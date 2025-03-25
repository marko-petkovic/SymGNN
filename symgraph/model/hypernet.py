import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter_mean

from symgraph.data_utils import letter_to_index as unique_letters

class HyperNodeWyckoff(nn.Module):

    def __init__(self, lin_in, lin_out, bias, hidden, *args, **kwargs):
        
        super().__init__()

        self.emb_letter = nn.Embedding(len(unique_letters), hidden)
        self.emb_mult = nn.Linear(1, hidden, bias=False)
        self.layer_emb = nn.Linear(1, hidden, bias=False)


        self.scale_weight = nn.Sequential(
            nn.Linear(3*hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, lin_in),
        )

        self.weight = nn.Parameter(torch.Tensor(1, lin_in, lin_out))
        self.bias = nn.Parameter(torch.Tensor(1, 1, lin_out)) if bias else None

        
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Perform hyperfan-in initialization of the weights and zeros the bias
        '''
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

        
    def forward(self, layer_idx, letter, number, *args, **kwargs):
        number = torch.log(number + 1).unsqueeze(-1)
        number_emb = self.emb_mult(number)
        letter_emb = self.emb_letter(letter)
        layer_emb = self.layer_emb(layer_idx)
        
        emb = torch.cat([number_emb, letter_emb, layer_emb], dim=-1)

        scale = self.scale_weight(emb).unsqueeze(-1)

        weights = scale * self.weight
        bias = self.bias.repeat(weights.size(0), 1, 1) if self.bias is not None else None

        return weights, bias
    

class HyperEdgeWyckoff(nn.Module):

    def __init__(self, lin_in, lin_out, bias, hidden, *args, **kwargs):
        
        super().__init__()

        self.emb_letter1 = nn.Embedding(len(unique_letters), hidden)
        self.emb_mult1 = nn.Linear(1, hidden, bias=False)

        self.emb_letter2 = nn.Embedding(len(unique_letters), hidden)
        self.emb_mult2 = nn.Linear(1, hidden, bias=False)
        
        self.layer_emb = nn.Linear(1, hidden, bias=False)

        self.scale_weight = nn.Sequential(
            nn.Linear(5*hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, lin_in),
        )

        self.weight = nn.Parameter(torch.Tensor(1, lin_in, lin_out))
        self.bias = nn.Parameter(torch.Tensor(1, 1, lin_out)) if bias else None

        
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Perform hyperfan-in initialization of the weights and zeros the bias
        '''

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias
        )

        
    def forward(self, layer_idx, letter1, number1, letter2, number2, *args, **kwargs):

        number1 = torch.log(number1 + 1).unsqueeze(-1)
        number2 = torch.log(number2 + 1).unsqueeze(-1)

        number_emb1 = self.emb_mult1(number1)
        letter_emb1 = self.emb_letter1(letter1)

        number_emb2 = self.emb_mult2(number2)
        letter_emb2 = self.emb_letter2(letter2)

        layer_emb = self.layer_emb(layer_idx)
        
        emb = torch.cat([number_emb1, letter_emb1, number_emb2, letter_emb2, layer_emb], dim=-1)

        scale = self.scale_weight(emb).unsqueeze(-1)
        # print(scale.shape, self.weight.shape)
        weights = scale * self.weight
        bias = self.bias.repeat(weights.size(0), 1, 1) if self.bias is not None else None

        return weights, bias



class HyperFiLMWyckoff(nn.Module):

    def __init__(self, lin_in, lin_out, bias, hidden, *args, **kwargs):
        
        super().__init__()

        self.emb_letter = nn.Embedding(len(unique_letters), hidden)
        self.emb_mult = nn.Linear(1, hidden, bias=False)
        self.layer_emb = nn.Linear(1, hidden, bias=False)

        self.film = nn.Sequential(
            nn.Linear(3*hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2*lin_out),
        )
        
        # self.reset_parameters()


    def forward(self, layer_idx, letter, number, *args, **kwargs):
        number = torch.log(number + 1).unsqueeze(-1)
        number_emb = self.emb_mult(number)
        letter_emb = self.emb_letter(letter)
        layer_emb = self.layer_emb(layer_idx)
        
        emb = torch.cat([number_emb, letter_emb, layer_emb], dim=-1)

        film = self.film(emb)

        gamma, beta = film.chunk(2, dim=-1)

        return gamma, beta


class HyperFiLMGenSet(nn.Module):

    def __init__(self, lin_in, lin_out, bias, hidden, *args, **kwargs):
        
        super().__init__()

        self.emb = nn.Linear(12, hidden, bias=False)
        # self.lin = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU())

        self.lin_out1 = nn.Linear(hidden, hidden)
        # self.lin_out2 = nn.Linear(hidden, hidden)

        self.film = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2*lin_out),
        )
        
        # self.ln1 = nn.LayerNorm(hidden)
        # self.ln2 = nn.LayerNorm(hidden)

        # self.reset_parameters()
    def forward(self, layer_idx, gen, gen_idx, i, *args, **kwargs):

        gen_emb = self.emb(gen)
        # gen_emb = self.lin(gen_emb)

        # gen_emb = F.elu((self.lin_out1(gen_emb)))

        gen_agg = scatter_mean(gen_emb, gen_idx, dim=0)

        # gen_agg = F.elu((self.lin_out2(gen_agg)))

        film = self.film(gen_agg)

        film = film[i]

        gamma, beta = film.chunk(2, dim=-1)

        return gamma, beta

class HyperFiLMGen(nn.Module):

    def __init__(self, lin_in, lin_out, bias, hidden, heads, emb_both=False, *args, **kwargs):
        
        super().__init__()

        self.emb_both = emb_both
        self.emb = nn.Linear(12, hidden, bias=False)
        self.lin = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU())
        
        if self.emb_both:
            self.emb2 = nn.Linear(12, hidden, bias=False)
            self.lin2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU())
        
        
        self.mha = nn.MultiheadAttention(hidden, heads, batch_first=True)

        self.lin_out1 = nn.Linear(hidden, hidden)
        self.lin_out2 = nn.Linear(hidden, hidden)

        self.film = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2*lin_out),
        )
        
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

        # self.reset_parameters()


    def forward(self, layer_idx, gen_i, gen_j, att_mask, *args, **kwargs):

        # TODO: something positional
        
        gen_i_emb = self.emb(gen_i)
        gen_i_emb = self.lin(gen_i_emb)

        if self.emb_both:
            gen_j_emb = self.emb2(gen_j)
            gen_j_emb = self.lin2(gen_j_emb)
        else:
            gen_j_emb = self.emb(gen_j)
            gen_j_emb = self.lin(gen_j_emb)

        att_mask = att_mask.repeat_interleave(self.mha.num_heads, dim=0)
        att, _ = self.mha(gen_i_emb, gen_j_emb, gen_j_emb, attn_mask=att_mask)

        agg_mask = (gen_j != -10)[...,0].unsqueeze(-1).repeat(1, 1, att.size(-1))
        
        # att[~agg_mask] = 0
        att = F.elu(self.ln1(self.lin_out1(att)))
        
        att[~agg_mask] = 0
        att = att.sum(dim=1)/agg_mask.sum(dim=1) 
        
        emb = F.elu(self.ln2(self.lin_out2(att)))
        
        film = self.film(emb)

        gamma, beta = film.chunk(2, dim=-1)

        return gamma, beta