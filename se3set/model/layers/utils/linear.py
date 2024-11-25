import torch

from e3nn import o3

from .tensor_product import FullyConnectedTensorProductRescale

class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        super().__init__(irreps_in1=irreps_in, 
                         irreps_in2=o3.Irreps('1x0e'), 
                         irreps_out=irreps_out, 
                         bias=bias, 
                         rescale=rescale, 
                         internal_weights=True, 
                         shared_weights=True, 
                         normalization=None)
    
    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out