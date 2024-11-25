import torch
from torch import nn
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter
from e3nn import o3

from typing import Sequence, Optional, Union, Dict, Any

from .layers import *
from .layers.utils import *
from .layers.config import _RESCALE, _DEFAULT_ATOM_TYPES, _MAX_ATOM_Z, _AVG_NUM_NODES, _AVG_NODE_DEGREE, _AVG_HYPEREDGE_DEGREE, _HYPEREDGE_CHANNEL


class SE3Set(nn.Module):
    """ An Equiformer version of `SetGNN` in `AllSet`.
    Refer to https://github.com/jianhao2016/AllSet/blob/main/src/models.py#L295-L484 and 
    https://github.com/atomicarchitects/equiformer/blob/master/nets/graph_attention_transformer_md17.py.  """

    def __init__(
        self, 
        irreps_out='1x0e',
        irreps_sh='1x0e+1x1e+1x2e', 
        max_radius: float = 5.0, 
        number_of_basis: int = 128, 
        rbf_basis_type: str = 'gaussian', 
        irreps_node_embedding='128x0e+64x1e+32x2e',
        irreps_hyperedge_embedding='128x0e+64x1e+32x2e',
        he_as_node_init: bool = False,
        use_he_as_message: bool = True,
        use_he_scalar: bool = True,
        use_he_degree_embedding: bool = True,
        num_layers: int = 6, 
        irreps_head='32x0e+16x1e+8x2e', 
        num_heads: int = 4, 
        rescale_degree: bool = False, 
        nonlinear_message: bool = False, 
        irreps_mlp_mid='128x0e+64x1e+32x2e',
        irreps_pre_attn=None,  
        fc_neurons: Optional[Sequence[int]] = [64, 64], 
        update_he_feature: bool = True,
        irreps_node_feature='512x0e',
        irreps_hyperedge_feature=None, 
        use_attn_head: bool = False, 
        norm_layer: str = 'layer', 
        alpha_drop: float = 0.2, 
        proj_drop: float = 0.0, 
        out_drop: float = 0.0, 
        drop_path_rate: float = 0.0, 
        mean_out: bool = False,
        scale=None, 
        atomref=None,
        dataset_statistics: Optional[Dict[str, Union[float, Any]]] = None,
        output_force: bool = False,
    ):  
        super(SE3Set, self).__init__()
        default_dataset_statistics = {'atom_types': _DEFAULT_ATOM_TYPES, 
                                      'max_atom_z': _MAX_ATOM_Z, 
                                      'avg_num_nodes': _AVG_NUM_NODES, 
                                      'avg_node_degree': _AVG_NODE_DEGREE, 
                                      'avg_hyperedge_degree': _AVG_HYPEREDGE_DEGREE, 
                                      'hyperedge_channel': _HYPEREDGE_CHANNEL}
        if dataset_statistics is None:
            dataset_statistics = default_dataset_statistics
        else:
            for k, v in default_dataset_statistics.items():
                if k not in dataset_statistics:
                    dataset_statistics[k] = v
        self.onehot_atom_types = len(dataset_statistics['atom_types'])
        self.atom_z_2_atom_types = -torch.ones(dataset_statistics['max_atom_z']+1, dtype=torch.long)
        self.atom_z_2_atom_types[dataset_statistics['atom_types']] = torch.arange(self.onehot_atom_types, dtype=torch.long)

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.use_attn_head = use_attn_head
        self.norm_layer = norm_layer
        self.scale = scale

        self.register_buffer('atomref', atomref)

        self.irreps_node_output = o3.Irreps(irreps_out)
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_hyperedge_embedding = o3.Irreps(irreps_hyperedge_embedding)
        self.lmax = self.irreps_node_embedding.lmax

        self.irreps_node_feature = o3.Irreps(irreps_node_feature) if irreps_node_feature is not None else self.irreps_node_embedding
        self.num_layers = num_layers
        self.irreps_hyperedge_feature = o3.Irreps(irreps_hyperedge_feature) if irreps_hyperedge_feature is not None else self.irreps_hyperedge_embedding

        self.num_layers = num_layers
        
        self.irreps_edge_attr = o3.Irreps(irreps_sh) if irreps_sh is not None else o3.Irreps.spherical_harmonics(self.lmax)

        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
        self.rbf_basis_type = rbf_basis_type

        self.he_as_node_init = he_as_node_init
        self.update_he_feature = update_he_feature
        self.use_he_as_message = use_he_as_message
        self.use_he_scalar = use_he_scalar

        self.hyperedge_scalar_indices = get_l0_indices(self.irreps_hyperedge_feature)
        self.rbp_fc_neurons, self.he_fc_neurons = None, None
        if fc_neurons is not None:
            self.rbp_fc_neurons = [self.number_of_basis] + fc_neurons
            if use_he_scalar:
                self.he_fc_neurons = [get_mul_0(self.irreps_hyperedge_embedding)] + fc_neurons 

        self.rbf : nn.Module = None
        if self.rbf_basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.rbf_basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius, rbf={'name': 'spherical_bessel'})
        elif self.rbf_basis_type == 'exp':
            self.rbf = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=self.max_radius, num_rbf=self.number_of_basis, trainable=False)
        elif self.rbf_basis_type == 'exp_noenvelope':
            self.rbf = ExpBasis(cutoff_lower=0.0, cutoff_upper=self.max_radius, num_rbf=self.number_of_basis, trainable=False)
        else:
            raise ValueError('Invalid radial basis type: {}'.format(self.rbf_basis_type))
        
        self.node_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, self.onehot_atom_types)
        if self.he_as_node_init:
            self.hyperedge_embed = NodeEmbeddingNetwork(self.irreps_hyperedge_embedding, self.onehot_atom_types)
        else:
            self.hyperedge_embed = HyperedgeEmbeddingNetwork(self.irreps_hyperedge_embedding, 
                                                             self.irreps_edge_attr, 
                                                             dataset_statistics['avg_hyperedge_degree'],
                                                             dataset_statistics['hyperedge_channel'], 
                                                             self.rbp_fc_neurons)

        self.node_deg_embed = DegreeEmbeddingNetwork(self.irreps_node_embedding, 
                                                     self.irreps_hyperedge_feature, 
                                                     dataset_statistics['avg_node_degree'], 
                                                     use_src_scalar=self.use_he_scalar, 
                                                     fc_neurons=self.he_fc_neurons)
        self.he_deg_embed = None
        if use_he_degree_embedding:
            if self.he_as_node_init:
                self.he_deg_embed = DegreeEmbeddingNetwork(self.irreps_hyperedge_embedding, 
                                                           self.irreps_edge_attr, 
                                                           dataset_statistics['avg_hyperedge_degree'], 
                                                           use_src_scalar=True,
                                                           fc_neurons=self.rbp_fc_neurons)
            else:
                self.he_deg_embed = DegreeEmbeddingNetwork(self.irreps_hyperedge_embedding, 
                                                           self.irreps_node_embedding, 
                                                           dataset_statistics['avg_hyperedge_degree'], 
                                                           use_src_scalar=True,
                                                           fc_neurons=self.rbp_fc_neurons)
            
        self.norm_node = get_norm_layer(self.norm_layer)(self.irreps_node_embedding)
        self.norm_he = get_norm_layer(self.norm_layer)(self.irreps_hyperedge_embedding)

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_block_node_output = self.irreps_node_embedding
                irreps_block_he_output = self.irreps_hyperedge_embedding
            else:
                irreps_block_node_output = self.irreps_node_feature
                irreps_block_he_output = self.irreps_hyperedge_feature
            
            blk = TransBlock(irreps_node_input=self.irreps_node_embedding, 
                             irreps_he_input=self.irreps_hyperedge_embedding, 
                             irreps_node_output=irreps_block_node_output, 
                             irreps_he_output=irreps_block_he_output,
                             irreps_edge_sh=self.irreps_edge_attr,
                             irreps_head=self.irreps_head, 
                             num_heads=self.num_heads,
                             use_he_as_message=self.use_he_as_message,
                             use_he_scalar=self.use_he_scalar,
                             node_fc_neurons=self.rbp_fc_neurons,
                             he_fc_neurons=self.he_fc_neurons,  
                             irreps_pre_attn=self.irreps_pre_attn, 
                             rescale_degree=self.rescale_degree, 
                             nonlinear_message=self.nonlinear_message, 
                             alpha_drop=self.alpha_drop, 
                             proj_drop=self.proj_drop, 
                             drop_path_rate=self.drop_path_rate, 
                             irreps_mlp_mid=self.irreps_mlp_mid, 
                             norm_layer=self.norm_layer)

            self.blocks.append(blk)
        
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_node_feature, self.out_drop)
        
        if self.use_attn_head:
            self.head = GraphAttention(irreps_dst_input=self.irreps_node_feature, 
                                       irreps_src_input=None,
                                       irreps_src_attr=self.irreps_hyperedge_feature, 
                                       irreps_dst_output=self.irreps_node_output,
                                       irreps_head=self.irreps_head, 
                                       num_heads=self.num_heads, 
                                       use_src_scalar=self.use_he_scalar,
                                       fc_neurons=self.he_fc_neurons,
                                       irreps_pre_attn=self.irreps_pre_attn, 
                                       rescale_degree=self.rescale_degree, 
                                       nonlinear_message=self.nonlinear_message,
                                       alpha_drop=alpha_drop, 
                                       proj_drop=proj_drop)
        else:
            assert self.irreps_node_feature.lmax == 0, 'If not using attention head, the output irreps must be scalar.'
            self.head = nn.Sequential(LinearRS(self.irreps_node_feature, self.irreps_node_feature, rescale=_RESCALE), 
                                      Activation(self.irreps_node_feature, acts=[nn.SiLU()]), 
                                      LinearRS(self.irreps_node_feature, self.irreps_node_output, rescale=_RESCALE))
        self.mean_out = mean_out
        if not mean_out:
            self.scaled_scatter = ScaledScatter(dataset_statistics['avg_num_nodes'])

        self.apply(self._init_weights)

        self.output_force = output_force

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]

        for module_name, module in self.named_modules():
            if (isinstance(module, nn.Linear) 
                or isinstance(module, nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer) 
                or isinstance(module, RadialBasis)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) and 'weight' in parameter_name:
                        continue

                    global_parameter_name = module_name + '.' + parameter_name

                    assert global_parameter_name in named_parameters_list
                    
                    no_wd_list.append(global_parameter_name)
                    
        return set(no_wd_list)
    
    def forward(self, data):
        atom_z, pos, v2e_index, e2v_index, batch = data.z, data.pos, data.v2e_index, data.e2v_index, data.batch

        if self.output_force:
            pos.requires_grad_(True)

        node_atom = self.atom_z_2_atom_types.to(atom_z.device)[atom_z]

        batch_node, batch_he = batch, (torch.arange(data.num_graphs).to(node_atom.device)).repeat_interleave(data.num_nw_hyperedges).long()

        # edge processing
        edge_index = torch.stack([e2v_index[1][v2e_index[1]], v2e_index[0]])
        edge_index, v2e_index_t = remove_self_loops(edge_index, v2e_index.T)
        v2e_index = v2e_index_t.T
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr, x=edge_vec, normalize=True, normalization='component')
        edge_length = torch.norm(edge_vec, dim=-1)
        edge_scalar = self.rbf(edge_length)

        # Node Embedding
        node_embedding = self.node_embed(node_atom)
        if self.he_as_node_init:
            # He Self Embedding
            he_embedding = self.hyperedge_embed(node_atom)
            he_embedding = he_embedding[e2v_index[1]]
            # He Degree Embedding
            if self.he_deg_embed is not None:
                he_degree_embedding = self.he_deg_embed(he_embedding, edge_sh, v2e_index[1], edge_scalar)
                hyperedge_features = he_embedding + he_degree_embedding
            else:
                hyperedge_features = he_embedding
        else:
            # He Self Embedding
            he_embedding = self.hyperedge_embed(edge_sh, edge_scalar, v2e_index, e2v_index)
            # He Degree Embedding
            if self.he_deg_embed is not None:
                he_degree_embedding = self.he_deg_embed(he_embedding, node_embedding[v2e_index[0]], v2e_index[1], edge_scalar)
                hyperedge_features = he_embedding + he_degree_embedding
            else:
                hyperedge_features = he_embedding
        # He Normalization
        sc_he = hyperedge_features
        hyperedge_features = self.norm_he(hyperedge_features, batch=batch_he)
        # Node Degree Embedding
        he_scalar_embedding = get_l0_as_scalar(hyperedge_features, self.hyperedge_scalar_indices) if self.use_he_scalar else None
        node_degree_embedding = self.node_deg_embed(node_embedding, hyperedge_features[e2v_index[0]], e2v_index[1], he_scalar_embedding)
        node_features = node_embedding + node_degree_embedding
        # Node Normalization
        sc_node = node_features
        node_features = self.norm_node(node_features, batch=batch_node)


        # Blocks
        if self.update_he_feature:    
            for blk in self.blocks:
                node_features, hyperedge_features, sc_node, sc_he = blk(node_input=node_features, 
                                                                        he_input=hyperedge_features,
                                                                        shortcut_node=sc_node,
                                                                        shortcut_he=sc_he, 
                                                                        edge_sh=edge_sh,
                                                                        edge_scalar=edge_scalar,
                                                                        v2e_index=v2e_index,
                                                                        e2v_index=e2v_index,
                                                                        batch_node=batch_node,
                                                                        batch_he=batch_he)
        else:
            for blk in self.blocks:
                node_features, _, sc_node, sc_he = blk(node_input=node_features, 
                                                       he_input=hyperedge_features,
                                                       shortcut_node=sc_node,
                                                       shortcut_he=sc_he, 
                                                       edge_sh=edge_sh,
                                                       edge_scalar=edge_scalar,
                                                       v2e_index=v2e_index,
                                                       e2v_index=e2v_index,
                                                       batch_node=batch_node,
                                                       batch_he=batch_he)

        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)

        if self.use_attn_head:
            outputs = self.head(dst_input=node_features, 
                                dst_index=e2v_index[1],
                                src_attr=hyperedge_features[e2v_index[0]])
        else:
            outputs = self.head(node_features)

        if self.mean_out:
            outputs = scatter(outputs, batch_node, dim=0, reduce='mean')
        else:
            outputs = self.scaled_scatter(outputs, batch_node, dim=0)
        
        if self.scale is not None:
            outputs = self.scale * outputs

        if self.output_force:
            # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L321-L328
            forces = -1 * (
                        torch.autograd.grad(
                            outputs,
                            pos,
                            grad_outputs=torch.ones_like(outputs),
                            create_graph=True,
                        )[0]
                    )
            return outputs, forces
        else:
            return outputs, None
    