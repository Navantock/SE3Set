import torch
from torch_scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

import torch_geometric

from .utils import *


@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
                 irreps_dst_input, 
                 irreps_src_input,
                 irreps_src_attr, 
                 irreps_dst_output,
                 irreps_head, 
                 num_heads, 
                 use_src_scalar: bool,
                 fc_neurons=None,
                 irreps_pre_attn=None, 
                 rescale_degree=False, 
                 nonlinear_message=False,
                 alpha_drop=0.1, 
                 proj_drop=0.1):
        
        super().__init__()
        self.use_src_scalar = use_src_scalar

        self.irreps_dst_input = o3.Irreps(irreps_dst_input)
        self.irreps_src_input = o3.Irreps(irreps_src_input) if irreps_src_input is not None else None
        self.irreps_src_attr = o3.Irreps(irreps_src_attr)
        self.irreps_toupdate_output = o3.Irreps(irreps_dst_output)
        self.irreps_pre_attn = self.irreps_dst_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        
        # Message From Dst
        self.merge_dst = LinearRS(self.irreps_dst_input, self.irreps_pre_attn, bias=False)
        self.merge_src = LinearRS(self.irreps_src_input, self.irreps_pre_attn, bias=False) if self.irreps_src_input is not None else None
        
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads)
        irreps_attn_heads = irreps_attn_heads.simplify() 
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()
        
        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn, 
                                         self.irreps_src_attr, 
                                         self.irreps_pre_attn, 
                                         fc_neurons=fc_neurons, 
                                         use_activation=True, 
                                         norm_layer=None, 
                                         internal_weights=False if self.use_src_scalar else True)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn, 
                                           self.irreps_src_attr, 
                                           irreps_attn_heads, 
                                           fc_neurons=None, 
                                           use_activation=False, 
                                           norm_layer=None, 
                                           internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn, 
                                     self.irreps_src_attr, 
                                     irreps_attn_all, 
                                     fc_neurons=fc_neurons, 
                                     use_activation=False, 
                                     norm_layer=None,
                                     internal_weights=False if self.use_src_scalar else True)
            self.vec2heads = Vec2AttnHeads((o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), num_heads)
        
        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        self.proj = LinearRS(irreps_attn_heads, self.irreps_toupdate_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_dst_input, drop_prob=proj_drop)
        
        
    def forward(self, dst_input, src_input, dst_index, src_index, src_attr, scalars=None):
        '''
            dst_input: [N_dst, irreps_dst_input.dim]
            dst_index: [E]
            src_attr: [E, irreps_src_attr.dim]
        '''
        message_dst = self.merge_dst(dst_input)
        message = message_dst[dst_index]
        if src_input is not None:
            message_src = self.merge_src(src_input)
            message = message + message_src[src_index]
        
        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(scalars) if scalars is not None else None
            message = self.sep_act.dtp(message, src_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=src_attr, edge_scalars=scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=src_attr, edge_scalars=scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, dst_index)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=dst_index, dim=0, dim_size=dst_input.shape[0])
        attn = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree = torch_geometric.utils.degree(dst_index, num_nodes=dst_input.shape[0], dtype=dst_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree
            
        node_output = self.proj(attn)
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str
