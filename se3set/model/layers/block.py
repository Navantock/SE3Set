import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode


from .utils import get_l0_indices, get_l0_as_scalar, get_norm_layer, FullyConnectedTensorProductRescale, GraphDropPath
from .graph_attn import GraphAttention
from .config import _RESCALE
from .feedforward import FeedForwardNetwork


@compile_mode('script')
class TransBlock(nn.Module):
    ''' 
    Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/graph_attention_transformer.py#L574C1-L667C27
    1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    '''
    
    def __init__(self,
                 irreps_node_input,
                 irreps_he_input,
                 irreps_node_output,
                 irreps_he_output,
                 irreps_edge_sh,
                 irreps_head, 
                 num_heads,
                 use_he_as_message: bool,
                 use_he_scalar: bool,
                 node_fc_neurons=None,
                 he_fc_neurons=None,
                 irreps_pre_attn=None, 
                 rescale_degree=False, 
                 nonlinear_message=False,
                 alpha_drop=0.1, 
                 proj_drop=0.1,
                 drop_path_rate=0.0,
                 irreps_mlp_mid=None,
                 norm_layer='layer'):
        
        super().__init__()

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_he_input = o3.Irreps(irreps_he_input)
        self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)
        self.hyperedge_scalar_indices = get_l0_indices(self.irreps_he_input)
        self.use_he_as_message = use_he_as_message
        self.use_he_scalar = use_he_scalar
        # output irreps for ffn
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_he_output = o3.Irreps(irreps_he_output)

        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None else self.irreps_node_input
        
        self.v2e_ga = GraphAttention(irreps_dst_input=self.irreps_he_input, 
                                     irreps_src_input=self.irreps_node_input,
                                     irreps_src_attr=self.irreps_edge_sh, 
                                     irreps_dst_output=self.irreps_he_output,
                                     irreps_head=self.irreps_head, 
                                     num_heads=self.num_heads, 
                                     use_src_scalar=True,
                                     fc_neurons=node_fc_neurons,
                                     irreps_pre_attn=self.irreps_pre_attn, 
                                     rescale_degree=self.rescale_degree, 
                                     nonlinear_message=self.nonlinear_message,
                                     alpha_drop=alpha_drop, 
                                     proj_drop=proj_drop)
        self.norm_he = get_norm_layer(norm_layer)(self.irreps_he_output)
        self.norm_he_1 = get_norm_layer(norm_layer)(self.irreps_he_input)
        self.ffn_he = FeedForwardNetwork(irreps_input=self.irreps_he_input,
                                         irreps_output=self.irreps_he_output, 
                                         irreps_mlp_mid=self.irreps_mlp_mid,
                                         proj_drop=proj_drop)
        self.norm_he_2 = get_norm_layer(norm_layer)(self.irreps_he_output)
        
        e2v_irreps_src_input = self.irreps_he_input if self.use_he_as_message else None
        e2v_irreps_src_attr = o3.Irreps('1x0e') if self.use_he_as_message else self.irreps_he_input
        e2v_use_src_scalar = False if self.use_he_as_message else self.use_he_scalar
        self.e2v_ga = GraphAttention(irreps_dst_input=self.irreps_node_input,
                                     irreps_src_input=e2v_irreps_src_input, 
                                     irreps_src_attr=e2v_irreps_src_attr, 
                                     irreps_dst_output=self.irreps_node_input,
                                     irreps_head=self.irreps_head, 
                                     num_heads=self.num_heads, 
                                     use_src_scalar=e2v_use_src_scalar,
                                     fc_neurons=he_fc_neurons,
                                     irreps_pre_attn=self.irreps_pre_attn, 
                                     rescale_degree=self.rescale_degree, 
                                     nonlinear_message=self.nonlinear_message,
                                     alpha_drop=alpha_drop, 
                                     proj_drop=proj_drop)
        self.norm_node_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ffn_node = FeedForwardNetwork(irreps_input=self.irreps_node_input,
                                           irreps_output=self.irreps_node_output, 
                                           irreps_mlp_mid=self.irreps_mlp_mid,
                                           proj_drop=proj_drop)
        self.norm_node_2 = get_norm_layer(norm_layer)(self.irreps_node_output)
        
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else None

        self.ffn_shortcut_node, self.ffn_shortcut_he = None, None
        if self.irreps_he_input != self.irreps_he_output:
            self.ffn_shortcut_he = FullyConnectedTensorProductRescale(self.irreps_he_input, 
                                                                      o3.Irreps("1x0e"), 
                                                                      self.irreps_he_output, 
                                                                      bias=True, 
                                                                      rescale=_RESCALE)
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut_node = FullyConnectedTensorProductRescale(self.irreps_node_input, 
                                                                        o3.Irreps("1x0e"), 
                                                                        self.irreps_node_output, 
                                                                        bias=True, 
                                                                        rescale=_RESCALE)
            
    def forward(self, node_input, he_input, shortcut_node, shortcut_he,
                edge_sh, edge_scalar, v2e_index, e2v_index, batch_node, batch_he, **kwargs):    
        output_node, output_he = shortcut_node, shortcut_he
        node_features, he_features = node_input, he_input

        # he: V2E_GraphAttention -> LayerNorm -> FeedForwardNetwork -> LayerNorm 
        # node: E2V_GraphAttention -> LayerNorm -> FeedForwardNetwork -> LayerNorm
        # V2E_GraphAttention -> FeedForwardNetwork -> E2V_GraphAttention -> FeedForwardNetwork

        # V2E
        he_features = self.v2e_ga(dst_input=he_features, 
                                  src_input=node_features,
                                  dst_index=v2e_index[1],
                                  src_index=v2e_index[0],
                                  src_attr=edge_sh,
                                  scalars=edge_scalar)
        if self.drop_path is not None:
            he_features = self.drop_path(he_features, batch_he)
        output_he = output_he + he_features
        #e2v_he_features = output_he
        he_features = self.norm_he_1(output_he, batch=batch_he)

        he_features = self.ffn_he(he_features)
        if self.ffn_shortcut_he is not None:
            output_he = self.ffn_shortcut_he(output_he, torch.ones_like(he_features.narrow(1, 0, 1)))
        if self.drop_path is not None:
            he_features = self.drop_path(he_features, batch=batch_he)
        output_he = output_he + he_features
        sc_he = output_he
        output_he = self.norm_he_2(output_he, batch=batch_he)

        # E2V
        he_features = output_he
        if self.use_he_as_message:
            node_features = self.e2v_ga(dst_input=node_features, 
                                        src_input=he_features,
                                        dst_index=e2v_index[1],
                                        src_index=e2v_index[0],
                                        src_attr=torch.ones_like(he_features[e2v_index[0]].narrow(1, 0, 1)),
                                        scalars=None)
        else:
            he_scalars = get_l0_as_scalar(he_features, self.hyperedge_scalar_indices) if self.use_he_scalar else None
            node_features = self.e2v_ga(dst_input=node_features, 
                                        src_input=None,
                                        dst_index=e2v_index[1],
                                        src_index=None,
                                        src_attr=he_features[e2v_index[0]],
                                        scalars=he_scalars)
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch_node)
        output_node = output_node + node_features
        node_features = self.norm_node_1(output_node, batch=batch_node)

        node_features = self.ffn_node(node_features)
        if self.ffn_shortcut_node is not None:
            output_node = self.ffn_shortcut_node(output_node, torch.ones_like(node_features.narrow(1, 0, 1)))
        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch_node)
        output_node = output_node + node_features
        sc_node = output_node
        output_node = self.norm_node_2(output_node, batch=batch_node)
        
        return output_node, output_he, sc_node, sc_he