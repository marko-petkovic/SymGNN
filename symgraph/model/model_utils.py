from torch import Tensor
from typing import Optional, List

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from itertools import chain

import numpy as np


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: Tensor) -> Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )
    

def pad_generators(generators, edge_index, padding_value=-10, batch_first=True):
    gens = list(chain.from_iterable(generators))
    gens = [torch.tensor(g, dtype=torch.float) for g in gens]
    gens = pad_sequence(gens, batch_first=batch_first, padding_value=padding_value).to(edge_index.device)

    i,j = edge_index

    gens_i, gens_j = gens[i], gens[j]

    # mask_i = (gens_i == padding_value)[...,0]
    mask_j = (gens_j == padding_value)[...,0]

    att_mask = mask_j.unsqueeze(1).repeat(1, gens_j.size(-2), 1)

    return gens_i, gens_j, att_mask


def set_generators(generators, device):

    gens = list(chain.from_iterable(generators))

    gen_len = torch.tensor([len(g) for g in gens], dtype=torch.long, device=device) # number of generators for each position
    gen_idx = torch.arange(len(gens), dtype=torch.long, device=device).repeat_interleave(gen_len) # index of each generator

    gens = torch.cat([torch.tensor(g, dtype=torch.float) for g in gens]).to(device)

    return gens, gen_idx


def langmuir_freundlich_2s(p, x):
    '''
    Langmuir-Freundlich isotherm with two sites

    p (torch.Tensor): parameters of the model
    x (torch.Tensor): input data (pressure)
    '''
    return p[:,0]*p[:,1]*x**p[:,2]/(1.0+p[:,1]*x**p[:,2])+p[:,3]*p[:,4]*x**p[:,5]/(1.0+p[:,4]*x**p[:,5])

def log_langmuir_freundlich_2s(p, x):
    '''
    Langmuir-Freundlich isotherm with two sites, where the pressure is log10-transformed

    p (torch.Tensor): parameters of the model
    x (torch.Tensor): input data (log10 pressure)
    '''
    # for numerical stability
    x_exp1 = 10**(x*p[:,2])
    x_exp2 = 10**(x*p[:,5])

    term1 = p[:,0]*p[:,1]*x_exp1/(1.0+p[:,1]*x_exp1)
    term2 = p[:,3]*p[:,4]*x_exp2/(1.0+p[:,4]*x_exp2)

    return term1 + term2
    
    # return p[:,0]*p[:,1]*(10**x)**p[:,2]/(1.0+p[:,1]*(10**x)**p[:,2])+p[:,3]*p[:,4]*(10**x)**p[:,5]/(1.0+p[:,4]*(10**x)**p[:,5])


    

def get_loading_and_derivative(params, p, iso_func='log_lf2'):

    if iso_func == 'lf2':
        iso = langmuir_freundlich_2s
    elif iso_func == 'log_lf2':
        iso = log_langmuir_freundlich_2s
    else:
        raise NotImplementedError
    
    # Clone and ensure p requires gradients for derivative calculation
    p = p.clone()
    p = p.repeat(1, params.shape[0]).detach().requires_grad_(True)


    q = iso(params, p)
    q_prime = torch.autograd.grad(q, p, grad_outputs=torch.ones_like(q), create_graph=True)[0]

    return q.T, q_prime.T


def get_loading_weight(epoch, warmup_epochs, total_epochs, max_weight=1.0):
    """
    Adjusts the weight for the loading loss dynamically.
    
    - Warmup period: weight is 0 (focus only on HOA).
    - After warmup: weight gradually increases to max_weight.
    
    Args:
        epoch (int): Current epoch.
        warmup_epochs (int): Number of warmup epochs.
        total_epochs (int): Total number of epochs.
        max_weight (float): Maximum weight for loading loss.

    Returns:
        float: Adjusted weight for loading loss.
    """
    if epoch < warmup_epochs:
        return 0.0  # No loading loss during warmup
    else:
        # Linear increase after warmup
        return max_weight * min((epoch - warmup_epochs) / (total_epochs - warmup_epochs), 1.0)
