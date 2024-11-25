import math
import torch
from torch import nn, Tensor
import numpy as np
from scipy.special import binom
from torch_geometric.nn.models.schnet import GaussianSmearing
from typing import Dict, Union


"""Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/gaussian_rbf.py
"""
@torch.jit.script
def gaussian(x, mean, std):
    a = (2 * math.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


# From Graphormer
class GaussianRadialBasisLayer(nn.Module):
    def __init__(self, num_basis, cutoff):
        super(GaussianRadialBasisLayer, self).__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff + 0.0
        self.mean = nn.Parameter(torch.zeros(1, self.num_basis))
        self.std = nn.Parameter(torch.zeros(1, self.num_basis))
        self.weight = nn.Parameter(torch.ones(1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1))
        
        self.std_init_max = 1.
        self.std_init_min = 1. / self.num_basis
        self.mean_init_max = 1.
        self.mean_init_min = 0
        nn.init.uniform_(self.mean, self.mean_init_min, self.mean_init_max)
        nn.init.uniform_(self.std, self.std_init_min, self.std_init_max)
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)
        

    def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None):
        x = dist / self.cutoff
        x = x.unsqueeze(-1)
        x = self.weight * x + self.bias
        x = x.expand(-1, self.num_basis)
        mean = self.mean
        std = self.std.abs() + 1e-5
        x = gaussian(x, mean, std)
        return x
    
    def extra_repr(self):
        return 'mean_init_max={}, mean_init_min={}, std_init_max={}, std_init_min={}'.format(
            self.mean_init_max, 
            self.mean_init_min, 
            self.std_init_max, 
            self.std_init_min
        )


"""Refer ro https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/gemnet/layers/radial_basis.py
"""
class PolynomialEnvelope(nn.Module):
    """Polynomial envelope function that ensures a smooth cutoff.

    Parameters
    ----------
    exponent: int
        Exponent of the envelope function.
    """
    def __init__(self, exponent: int):
        super(PolynomialEnvelope, self).__init__()

        assert exponent > 0
        
        self.p = float(exponent)
        self.a = - (self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = - self.p * (self.p + 1) / 2

    def forward(self, d_scaled: Tensor):
        env_val = 1 + self.a * d_scaled ** self.p + self.b * d_scaled ** (self.p + 1) + self.c * d_scaled ** (self.p + 2)
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(nn.Module):
    """Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """
    def __init__(self):
        super(ExponentialEnvelope, self).__init__()

    def forward(self, d_scaled: Tensor):
        env_val = torch.exp(- (d_scaled ** 2) / ((1 - d_scaled) * (1 + d_scaled)))
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class SphericalBesselBasis(nn.Module):
    """1D spherical Bessel basis

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """
    def __init__(self, num_radial: int, cutoff: float):
        super(SphericalBesselBasis, self).__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = nn.Parameter(
            data=torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)), 
            requires_grad=True
        )

    def forward(self, d_scaled: Tensor):
        return self.norm_const / d_scaled[:, None] * torch.sin(self.frequencies * d_scaled[:, None])  # (num_edges, num_radial)


class BernsteinBasis(nn.Module):
    """Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """
    def __init__(self, num_radial: int, pregamma_initial: float=0.45264):
        super(BernsteinBasis, self).__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer('prefactor', torch.tensor(prefactor, dtype=torch.float), persistent=False)

        self.pregamma = nn.Parameter(data=torch.tensor(pregamma_initial, dtype=torch.float), requires_grad=True)
        self.softplus = nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer('exp1', exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer('exp2', exp2[None, :], persistent=False)

    def forward(self, d_scaled: Tensor):
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)


class RadialBasis(nn.Module):
    """
    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """
    def __init__(
        self, 
        num_radial: int, 
        cutoff: float, 
        rbf: Dict[str, str] = {'name': 'gaussian'}, 
        envelope: Dict[str, Union[str, int]] = {'name': 'polynomial', 'exponent': 5},
    ):
        super(RadialBasis, self).__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope['name'].lower()
        env_hparams = envelope.copy()

        del env_hparams['name']

        if env_name == 'polynomial':
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == 'exponential':
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf['name'].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams['name']

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == 'gaussian':
            self.rbf = GaussianSmearing(start=0, stop=1, num_gaussians=num_radial, **rbf_hparams)
        elif rbf_name == 'spherical_bessel':
            self.rbf = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff, **rbf_hparams)
        elif rbf_name == 'bernstein':
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)  # (nEdges, num_radial)
    

"""Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/expnorm_rbf.py
"""
class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (torch.cos(math.pi * (2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 1.0)) + 1.0)

            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


# https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/utils.py#L111
class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2)

class ExpBasis(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(ExpBasis, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return torch.exp(-self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2)
   

"""Refer to https://github.com/atomicarchitects/equiformer/blob/master/nets/radial_func.py
"""
class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super(RadialProfile, self).__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue

            if (i == len(ch_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True

            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_biases))
            input_channels = ch_list[i]
            
            if i == len(ch_list) - 1:
                break
            
            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))

            modules.append(nn.SiLU())
        
        self.net = nn.Sequential(*modules)
        
        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.offset, -bound, bound)
        
    def forward(self, f_in):
        f_out = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1)
            
        return f_out