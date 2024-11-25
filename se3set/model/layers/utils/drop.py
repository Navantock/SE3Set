import torch
from torch import nn
from e3nn import o3


class EquivariantDropout(nn.Module):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/drop.py#L68-L86
    """
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps, o3.Irreps('{}x0e'.format(self.num_irreps)))
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out
    

class GraphDropPath(nn.Module):
    """Consider batch for graph data when dropping paths.
    Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/drop.py#L45C1-L64C53
    """
    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size, ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out
    
    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)
    

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/drop.py#L13C1-L28C18
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output