import torch
from torch import nn
from e3nn import o3
from e3nn.util.jit import compile_mode

from .activate import Activation


def irreps2gate(irreps):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/tensor_product_rescale.py#L177C1-L192C54
    """
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))

    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = '0e'
    else:
        ir = None
        
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


@compile_mode('script')
class Gate(nn.Module):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/fast_activation.py#L91-L160
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """
    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated):
        super(Gate, self).__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f'Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}')
        
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(f'Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}')
        
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(f'There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates')

        #assert len(irreps_scalars) == 1
        #assert len(irreps_gates) == 1

        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated
        self._irreps_in = (irreps_scalars + irreps_gates + irreps_gated).simplify()
        
        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def forward(self, features):
        scalars_dim = self.irreps_scalars.dim
        gates_dim = self.irreps_gates.dim
        input_dim = self.irreps_in.dim
        scalars = features.narrow(-1, 0, scalars_dim)
        gates = features.narrow(-1, scalars_dim, gates_dim)
        gated = features.narrow(-1, (scalars_dim + gates_dim), (input_dim - scalars_dim - gates_dim))
        
        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars

        return features

    @property
    def irreps_in(self):
        """Input representations.
        """
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations.
        """
        return self._irreps_out
    
    def __repr__(self):
        return f'{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})'