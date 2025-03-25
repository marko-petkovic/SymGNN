import torch
import torch.nn as nn

from symgraph.model.hypernet import (
    HyperNodeWyckoff,
    HyperEdgeWyckoff, 
    HyperFiLMWyckoff, 
    HyperFiLMGen,
    HyperFiLMGenSet,
)


class SymLinearNode(nn.Module):

    def __init__(
            self,
            in_features,
            out_features, 
            bias, 
            hyper_hidden, 
            hypernet_type='wyckoff',
            *args, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if hypernet_type == 'wyckoff':
            self.hyper = HyperNodeWyckoff(in_features, out_features, bias, hyper_hidden, *args, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, x, layer_idx, *args, **kwargs):
        
        weights, bias = self.hyper(layer_idx, *args, **kwargs)

        out = torch.matmul(x.unsqueeze(1), weights)

        if bias is not None:
            out = out + bias

        return out.squeeze(1)
    

class SymLinearEdge(nn.Module):

    def __init__(self, in_features, out_features, bias, hyper_hidden, *args, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.hyper = HyperEdgeWyckoff(in_features, out_features, bias, hyper_hidden, *args, **kwargs)

    def forward(self, x, layer_idx, *args, **kwargs):
        
        weights, bias = self.hyper(layer_idx, *args, **kwargs)
        out = torch.matmul(x.unsqueeze(1), weights)

        if bias is not None:
            out = out + bias

        return out.squeeze(1)
    

class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias, *args, **kwargs):
        super().__init__()
        

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, *args, **kwargs):
        return self.linear(x)
    


class FiLM(nn.Module):

    def __init__(self, in_features, out_features, bias, hyper_hidden, hypernet_type = 'wyckoff_film', *args, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if hypernet_type == 'wyckoff_film':
            self.hyper = HyperFiLMWyckoff(in_features, out_features, bias, hyper_hidden, *args, **kwargs)
        elif hypernet_type == 'gens_mha_film':
            self.hyper = HyperFiLMGen(in_features, out_features, bias, hyper_hidden, *args, **kwargs)
        elif hypernet_type == 'gens_set_film':
            self.hyper = HyperFiLMGenSet(in_features, out_features, bias, hyper_hidden, *args, **kwargs)
        else:
            raise NotImplementedError
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, layer_idx, *args, **kwargs):
        
        x = self.linear(x)

        gamma, beta = self.hyper(layer_idx, *args, **kwargs)

        out = gamma * x + beta

        return out