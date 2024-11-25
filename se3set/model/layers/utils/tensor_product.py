import torch
from torch import nn

import collections
from collections import defaultdict

from e3nn import o3
from e3nn.math import perm

from .activate import Activation
from .gate import Gate, irreps2gate

from ..config import _RESCALE


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
            
    return mul_0

def get_l0_slice(irreps):
    '''
        Get the index of 0e in irreps.
    '''
    l0_slices = []
    for (mul, ir), s in zip(irreps, irreps.slices()):
        if ir.l == 0 and ir.p == 1:
            l0_slices.append(s)
    return l0_slices

def get_l0_indices(irreps):
    '''
        Get the index of 0e in irreps.
    '''
    l0_indices = []
    start_idx = 0
    for (mul, ir) in irreps:
        if ir.l == 0 and ir.p == 1:
            l0_indices.append((start_idx, start_idx + mul))
        start_idx = start_idx + mul * ir.dim
    return l0_indices

def get_l0_as_scalar(x, l0_indices):
    out = []
    for ir_idx, (start_idx, end_idx) in enumerate(l0_indices):
        temp = x.narrow(1, start_idx, end_idx - start_idx)
        out.append(temp)
    return torch.cat(out, dim=-1)

def sort_irreps_even_first(irreps):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/tensor_product_rescale.py#L224-L231
    """
    Ret = collections.namedtuple('sort', ['irreps', 'p', 'inv'])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    inv_perm = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (l, -p)) for l, p, _, mul in out])
    return Ret(irreps, inv_perm, inv)


class TensorProductRescale(nn.Module):
    def __init__(self, 
                 irreps_in1, 
                 irreps_in2, 
                 irreps_out, 
                 instructions,
                 bias: bool = True, 
                 rescale: bool = True, 
                 internal_weights=None, 
                 shared_weights=None, 
                 normalization=None):
        
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.rescale = rescale
        self.use_bias = bias
        
        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(irreps_in1=self.irreps_in1, 
                                   irreps_in2=self.irreps_in2, 
                                   irreps_out=self.irreps_out,
                                   instructions=instructions, 
                                   normalization=normalization,
                                   internal_weights=internal_weights, 
                                   shared_weights=shared_weights,
                                   path_normalization='none')
        
        self.init_rescale_bias()
    
    
    def calculate_fan_in(self, ins):
        return {
            'uvw': (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            'uvu': self.irreps_in2[ins.i_in2].mul,
            'uvv': self.irreps_in1[ins.i_in1].mul,
            'uuw': self.irreps_in1[ins.i_in1].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]
        
        
    def init_rescale_bias(self) -> None:
        # For each zeroth order output irrep we need a bias
        # Store tuples of slices and corresponding biases in a list
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [l for (mul, (l, p)) in self.irreps_bias]
        self.irreps_bias_parity = [p for (mul, (l, p)) in self.irreps_bias]
        self.irreps_bias_dims = [mul for (mul, (l, p)) in self.irreps_bias]

        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if self.irreps_bias_orders[slice_idx] == 0 and self.irreps_bias_parity[slice_idx] == 1:
                    # non-pseudo scalar, add bias
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(torch.zeros(self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype))
                    self.bias.append(out_bias)
                    self.bias_slices.append(out_slice)
                    self.bias_slice_idx.append(slice_idx)
        self.bias = torch.nn.ParameterList(self.bias)
       

        # Determine the order for each output tensor and their dims
        irreps_out = self.irreps_out
        self.irreps_out_orders = [l for (mul, (l, p)) in irreps_out]
        self.irreps_out_dims = [mul for (mul, (l, p)) in irreps_out]
        self.irreps_out_slices = irreps_out.slices()
        self.slices_sqrt_k = {}

        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = defaultdict(int)  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                slices_fan_in[slice_idx] += self.calculate_fan_in(instr)
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)
                
            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                

    def forward_tp_rescale_bias(self, x, y, weight=None):
        out = self.tp(x, y, weight)
        
        if self.use_bias:
            for (_, slice, bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out
        

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(self,
                 irreps_in1, 
                 irreps_in2, 
                 irreps_out,
                 bias=True, 
                 rescale=True,
                 internal_weights=None, 
                 shared_weights=None,
                 normalization=None):
        
        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, 
                         irreps_in2, 
                         irreps_out, 
                         instructions=instructions,bias=bias, 
                         rescale=rescale,
                         internal_weights=internal_weights, 
                         shared_weights=shared_weights,
                         normalization=normalization)


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/graph_attention_transformer.py#L128C1-L154C19
    """
    def __init__(
        self, 
        irreps_in1, 
        irreps_in2, 
        irreps_out, 
        bias=True, 
        rescale=True, 
        internal_weights=None, 
        shared_weights=None, 
        normalization=None
    ):    
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, 
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, 
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )

        super(FullyConnectedTensorProductRescaleSwishGate, self).__init__(
            irreps_in1, 
            irreps_in2, 
            gate.irreps_in, 
            bias=bias, 
            rescale=rescale, 
            internal_weights=internal_weights, 
            shared_weights=shared_weights, 
            normalization=normalization
        )
        self.gate = gate
        
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    

def DepthwiseTensorProduct(
    irreps_central_input, 
    irreps_related_input, 
    irreps_node_output, 
    internal_weights=False, 
    bias=True
):
    """The irreps of output is pre-determined. 
    `irreps_central_output` is used to get certain types of vectors.
    Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/graph_attention_transformer.py#L157-L183
    """
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_central) in enumerate(irreps_central_input):
        for j, (_, ir_related) in enumerate(irreps_related_input):
            for ir_out in ir_central * ir_related:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)
    instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(
        irreps_central_input, 
        irreps_related_input,
        irreps_output, 
        instructions, 
        internal_weights=internal_weights, 
        shared_weights=internal_weights, 
        bias=bias, 
        rescale=_RESCALE
    )
    return tp