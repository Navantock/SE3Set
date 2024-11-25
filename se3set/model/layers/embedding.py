import torch
from torch import nn
import torch.nn.functional as F
from e3nn import o3
from typing import List

from .utils import (LinearRS, 
                    DepthwiseTensorProduct, 
                    RadialProfile, 
                    ScaledScatter)

from .config import _USE_BIAS, _RESCALE


class HyperedgeEmbeddingNetwork(nn.Module):
    def __init__(self, 
                 irreps_hyperedge_embedding, 
                 irreps_edge_attr, 
                 scale_scatter_num: int,
                 he_channel_num: int, 
                 fc_neurons: List, 
                 **kwargs) -> None:
        super(HyperedgeEmbeddingNetwork, self).__init__()
        self.irreps_hyperedge_embedding = o3.Irreps(irreps_hyperedge_embedding)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.he_channel_num = he_channel_num
        self.irreps_hyperedge_attr = self.irreps_edge_attr * self.he_channel_num
 
        self.rad = RadialProfile(fc_neurons + [self.he_channel_num])
        self.he_scatter = ScaledScatter(scale_scatter_num)
        self.lin_trans = LinearRS(self.irreps_hyperedge_attr, self.irreps_hyperedge_embedding, bias=_USE_BIAS, rescale=_RESCALE)

    def forward(self, edge_sh, edge_scalar, v2e_index, e2v_index):
        he_index = v2e_index[1]
        edge_weight = self.rad(edge_scalar)
        edge_sh_expand = edge_sh.unsqueeze(-2)
        edge_weight = edge_weight.unsqueeze(-1)
        he = edge_weight * edge_sh_expand
        he = he.flatten(start_dim=-2)
        he = self.he_scatter(he, he_index, dim=0, dim_size=e2v_index.shape[1])
        he = self.lin_trans(he)
        return he
        

class NodeEmbeddingNetwork(nn.Module):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/graph_attention_transformer.py#L670C1-L690C59
    """    
    def __init__(self, 
                 irreps_node_embedding, 
                 max_atom_type, 
                 bias=True):
        super(NodeEmbeddingNetwork, self).__init__()
        self.max_atom_type = max_atom_type

        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
                                      self.irreps_node_embedding, 
                                      bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
    def forward(self, node_atom):
        node_atom_onehot = F.one_hot(node_atom, self.max_atom_type).float()
        node_embedding = self.atom_type_lin(node_atom_onehot)
        return node_embedding


class DegreeEmbeddingNetwork(nn.Module):
    def __init__(self, 
                 irreps_dst_embedding, 
                 irreps_src_attr, 
                 avg_aggregate_num, 
                 use_src_scalar: bool,
                 fc_neurons=None):
        super(DegreeEmbeddingNetwork, self).__init__()
        self.irreps_dst_embedding = o3.Irreps(irreps_dst_embedding)
        self.irreps_src_attr = o3.Irreps(irreps_src_attr)
        self.lin = LinearRS(o3.Irreps('1x0e'), 
                            self.irreps_dst_embedding, 
                            bias=_USE_BIAS, 
                            rescale=_RESCALE)
        self.dtp = DepthwiseTensorProduct(self.irreps_dst_embedding, 
                                          self.irreps_src_attr, 
                                          self.irreps_dst_embedding, 
                                          internal_weights=False if use_src_scalar else True, 
                                          bias=False)
        self.rad = None
        if fc_neurons is not None:
            self.rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dtp.irreps_out.simplify(), irreps_dst_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, dst_input, src_attr, dst_index, scalars = None):
        dst_features = torch.ones_like(dst_input.narrow(1, 0, 1))
        dst_features = self.lin(dst_features)
        weight = self.rad(scalars) if scalars is not None and self.rad is not None else None
        deg_features = self.dtp(dst_features[dst_index], src_attr, weight)
        deg_features = self.proj(deg_features)
        dst_features = self.scale_scatter(deg_features, dst_index, dim=0, dim_size=dst_features.shape[0])
        return dst_features