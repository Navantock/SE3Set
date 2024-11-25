from torch import nn
import torch

from e3nn.util.jit import compile_mode
from e3nn import o3
from .utils import FullyConnectedTensorProductRescale, FullyConnectedTensorProductRescaleSwishGate, EquivariantDropout
from .config import _RESCALE


@compile_mode('script')
class FeedForwardNetwork(nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
                 irreps_input, 
                 irreps_output, 
                 irreps_mlp_mid=None,
                 proj_drop=0.1):
        
        super().__init__()
        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None else self.irreps_input
        self.irreps_output = o3.Irreps(irreps_output)
        
        self.lin_1 = FullyConnectedTensorProductRescaleSwishGate(self.irreps_input, o3.Irreps("1x0e"), self.irreps_mlp_mid, bias=True, rescale=_RESCALE)
        self.lin_2 = FullyConnectedTensorProductRescale(self.irreps_mlp_mid, o3.Irreps("1x0e"), self.irreps_output, bias=True, rescale=_RESCALE)
        
        self.proj_drop = EquivariantDropout(self.irreps_output, drop_prob=proj_drop) if proj_drop != 0.0 else None
        
    def forward(self, input, **kwargs):
        ones = torch.ones_like(input.narrow(1, 0, 1), device=input.device)
        node_output = self.lin_1(input, ones)
        node_output = self.lin_2(node_output, ones)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output