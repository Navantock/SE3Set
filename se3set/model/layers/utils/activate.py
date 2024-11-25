import torch
from torch import nn
from e3nn import o3
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode
from e3nn.util._argtools import _get_device


@compile_mode('trace')
class Activation(nn.Module):
    """Directly apply activation when irreps is type-0.
    Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/fast_activation.py#L15-L87
    """
    def __init__(self, irreps_in, acts):
        super(Activation, self).__init__()
        irreps_in = o3.Irreps(irreps_in)

        assert len(irreps_in) == len(acts), "irreps_in and activation should have same size, got irreps_in: \n {} \n acts: \n {}".format(irreps_in, acts)

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError('Activation: cannot apply an activation function to a non-scalar input.')

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError('Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.')
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = nn.ModuleList(acts)

    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + '{} -> {}, '.format(self.irreps_in, self.irreps_out)
        return output_str
    
    def forward(self, features: torch.Tensor, dim=-1):
        # directly apply activation without narrow
        if len(self.acts) == 1:
            return self.acts[0](features)
        
        output = []
        index = 0
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir.dim))
                
            index += mul * ir.dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        #else:
        #    return torch.zeros_like(features)


class SmoothLeakyReLU(nn.Module):
    """Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/graph_attention_transformer.py#L54-L67
    """
    def __init__(self, negative_slope=0.2):
        super(SmoothLeakyReLU, self).__init__()
        self.alpha = negative_slope
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)