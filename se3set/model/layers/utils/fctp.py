import torch
from torch import nn

from e3nn import o3

from .norm import get_norm_layer
from .tensor_product import DepthwiseTensorProduct
from .linear import LinearRS
from .gate import Gate, irreps2gate
from .activate import Activation
from .rbf import RadialProfile


class SeparableFCTP(nn.Module):
    '''
        Use separable FCTP for spatial convolution.
    '''
    def __init__(self, 
                 irreps_input4update, 
                 irreps_input4mp, 
                 irreps_output4update, 
                 fc_neurons, 
                 use_activation=False, 
                 norm_layer='graph', 
                 internal_weights=False):
        
        super().__init__()
        self.irreps_input4update = o3.Irreps(irreps_input4update)
        self.irreps_input4mp = o3.Irreps(irreps_input4mp)
        self.irreps_output4update = o3.Irreps(irreps_output4update)
        
        self.dtp = DepthwiseTensorProduct(self.irreps_input4update, 
                                          self.irreps_input4mp, 
                                          self.irreps_output4update,  
                                          internal_weights=internal_weights,
                                          bias=False)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k
                
        irreps_lin_output = self.irreps_output4update
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_output4update)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = get_norm_layer(norm_layer)(self.lin.irreps_out) if norm_layer is not None else None
        
        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_output4update, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input, edge_attr, edge_scalars=None, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        