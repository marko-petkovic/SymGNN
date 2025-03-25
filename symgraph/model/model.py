import torch
import torch.nn as nn

import torch.nn.functional as F

from symgraph.model.model_utils import RBFExpansion, pad_generators, set_generators, langmuir_freundlich_2s
from symgraph.model.symlinear import SymLinearNode, SymLinearEdge, Linear, FiLM

from typing import List

from torch_scatter import scatter_add, scatter_mean


class GNN(nn.Module):

    def __init__(self,
                 num_layers: int = 6, 
                 hidden: int = 64, 
                 bias: bool = False, 
                 out_size: int = int,
                 attention: bool = True, 
                 hyper_hidden: int = 64, 
                 residual=True, 
                 hyper_node_type='wyckoff', 
                 hyper_edge_type='wyckoff',
                 hyper_num_heads=4,
                 edge_dropout=0.0,
                 iso_dropout=0.0,
                 node_agg='mean',
                 graph_agg='mean',
                 *args, **kwargs):

        super().__init__()

        possible_hyper_edge = ['wyckoff', 'wyckoff_film', 'gens_mha_film', 'gens_set_film','none']
        possible_hyper_node = ['wyckoff', 'gens_set_film','none']

        possible_agg = ['mean', 'sum']

        self.edge_dropout = edge_dropout

        assert hyper_node_type in possible_hyper_node, f'hyper_node_type must be one of {possible_hyper_node}'
        assert hyper_edge_type in possible_hyper_edge, f'hyper_edge_type must be one of {possible_hyper_edge}'
        assert node_agg in possible_agg, f'node_agg must be one of {possible_agg}'
        assert graph_agg in possible_agg, f'graph_agg must be one of {possible_agg}'

        self.num_layers = num_layers
        self.hidden = hidden
        self.residual = residual

        self.node_agg = node_agg
        self.graph_agg = graph_agg

        self.hyper_node_type = hyper_node_type
        self.hyper_edge_type = hyper_edge_type
        self.attention = attention

        self.edge_emb = nn.Sequential(
            RBFExpansion(0., 10., 64),
            nn.Linear(64, hidden),
            nn.ELU(),
        )

        self.emb_ats = nn.Linear(1, hidden, bias=False)

        
        self.hyper_attention = False
        self.hyper_set = False
        self.hyper_set_node = False

        if self.hyper_node_type == 'wyckoff':
            self.node_update_1 = SymLinearNode(2*hidden, hidden, bias, hyper_hidden, self.hyper_node_type, *args, **kwargs)
        elif self.hyper_node_type == 'gens_set_film':
            self.node_update_1 = nn.ModuleList([FiLM(2*hidden, hidden, bias, hyper_hidden, self.hyper_node_type, *args, **kwargs) for _ in range(num_layers)])
            self.hyper_set_node = True
        elif self.hyper_node_type == 'none':
            self.node_update_1 = nn.ModuleList([Linear(2*hidden, hidden, bias, *args, **kwargs) for _ in range(num_layers)])

        self.node_update_2 = nn.ModuleList([Linear(hidden, hidden, bias, *args, **kwargs) for _ in range(num_layers)])

        self.node_ln_1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.node_ln_2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])


        if self.hyper_edge_type == 'wyckoff':
            self.edge_update_1 = SymLinearEdge(3*hidden, hidden, bias, hyper_hidden, self.hyper_edge_type, *args, **kwargs) 
        elif self.hyper_edge_type == 'wyckoff_film':
            self.edge_update_1 = nn.ModuleList([FiLM(3*hidden, hidden, bias, hyper_hidden, self.hyper_edge_type, *args, **kwargs) for _ in range(num_layers)])
        elif self.hyper_edge_type == 'gens_mha_film':
            self.edge_update_1 = nn.ModuleList([FiLM(3*hidden, hidden, bias, hyper_hidden, self.hyper_edge_type, heads=hyper_num_heads, emb_both=False, *args, **kwargs) for _ in range(num_layers)])
            self.hyper_attention = True
        elif self.hyper_edge_type == 'gens_set_film':
            self.edge_update_1 = nn.ModuleList([FiLM(3*hidden, hidden, bias, hyper_hidden, self.hyper_edge_type, *args, **kwargs) for _ in range(num_layers)])
            self.hyper_set = True
        elif self.hyper_edge_type == 'none':
            self.edge_update_1 = nn.ModuleList([Linear(3*hidden, hidden, bias) for _ in range(num_layers)])

        self.edge_update_2 = nn.ModuleList([Linear(hidden, hidden, bias, *args, **kwargs) for _ in range(num_layers)])
        
        self.edge_ln_1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.edge_ln_2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

        if self.attention:
            self.attention = nn.ModuleList([nn.Sequential(nn.Linear(hidden, 1)) for _ in range(num_layers)])

        self.out_layer = nn.Linear(hidden, out_size)

        self.iso_layer = nn.Sequential(nn.Linear(hidden + 2, hidden),
                                    #    nn.LayerNorm(hidden), 
                                       nn.ELU(),
                                       nn.Dropout(iso_dropout),
                                       nn.Linear(hidden, 1), nn.Softplus())    


    def forward(self, data, pres=None, integrate=False):

        x, edge_index, edge_attr, batch, wyck, mults = data.x, data.edge_index, data.edge_attr, data.batch, data.wyck, data.mults
        gens, gen_idx = data.gens, data.gen_index

        if self.training and self.edge_dropout > 0:
            edge_mask = torch.rand(edge_index.size(1)) > self.edge_dropout
            edge_index = edge_index[:, edge_mask]
            edge_attr = edge_attr[edge_mask]

        if self.hyper_attention:
            gens = data.gens
            gens_i, gens_j, att_mask = pad_generators(gens, edge_index)

        # elif self.hyper_set or self.hyper_set_node:
        #     gens = data.gens
        #     gens, gen_idx = set_generators(gens, x.device)



        x = self.emb_ats(x)
        edge_emb = self.edge_emb(edge_attr)

        for layer in range(self.num_layers):
            
            layer_idx_edge = torch.zeros((edge_index.size(1),1), dtype=torch.float, device=x.device) + layer
            layer_idx_node = torch.zeros((x.size(0),1), dtype=torch.float, device=x.device) + layer

            # calculate messages
            i,j = edge_index

            m_ij = torch.cat([x[i], x[j], edge_emb], dim=-1)

            if self.hyper_edge_type == 'wyckoff':
                # update edges
                m_ij = self.edge_update_1(m_ij, layer_idx_edge, wyck[i], mults[i], wyck[j], mults[j])
            elif self.hyper_edge_type == 'wyckoff_film':
                m_ij = self.edge_update_1[layer](m_ij, layer_idx_edge, wyck[i], mults[i])
            elif self.hyper_edge_type == 'gens_mha_film':
                m_ij = self.edge_update_1[layer](m_ij, layer_idx_edge, gens_i, gens_j, att_mask)
            elif self.hyper_edge_type == 'gens_set_film':
                m_ij = self.edge_update_1[layer](m_ij, layer_idx_edge, gens, gen_idx, i)
            elif self.hyper_edge_type == 'none':
                m_ij = self.edge_update_1[layer](m_ij)
            else:
                raise NotImplementedError
            m_ij = self.edge_ln_1[layer](m_ij)
            m_ij = F.elu(m_ij)

            m_ij = self.edge_update_2[layer](m_ij)
            m_ij = self.edge_ln_2[layer](m_ij)
            m_ij = F.elu(m_ij)

            # attention
            if self.attention:
                att = self.attention[layer](m_ij)
                m_ij = m_ij * att

            # aggregate messages
            if self.node_agg == 'mean':
                m_i = scatter_mean(m_ij, i, dim=0, dim_size=x.size(0))
            elif self.node_agg == 'sum':
                m_i = scatter_add(m_ij, i, dim=0, dim_size=x.size(0))
            else:
                raise NotImplementedError

            # update nodes
            h = torch.cat([x, m_i], dim=-1)

            if self.hyper_node_type == 'wyckoff':
                h = self.node_update_1(h, layer_idx_node, wyck, mults)
            elif self.hyper_node_type == 'gens_set_film':
                h = self.node_update_1[layer](h, layer_idx_node, gens, gen_idx, torch.arange(x.size(0), dtype=torch.long, device=x.device))
            elif self.hyper_node_type == 'none':
                h = self.node_update_1[layer](h)
            else:
                raise NotImplementedError

            h = self.node_ln_1[layer](h)
            h = F.elu(h)

            h = self.node_update_2[layer](h)
            
            h = self.node_ln_2[layer](h)
            h = F.elu(h)

            # residual
            if self.residual:
                h = h + x

            x = h

        if self.graph_agg == 'mean':
            x = scatter_mean(x, batch, dim=0)
        elif self.graph_agg == 'sum':
            x = scatter_add(x, batch, dim=0)
        else:
            raise NotImplementedError

        out = self.out_layer(x)


        
        p = pres.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = x.unsqueeze(1).repeat(1, p.shape[1], 1)
        h = out.unsqueeze(1).repeat(1, p.shape[1], 1)
        # print("Shapes of pres, x, hoa:")
        # print(p.shape, x.shape, h.shape)

        hidden = torch.cat([x, p, h], dim=-1)

        q_hat = self.iso_layer(hidden).squeeze(-1)

        # print("Shape of q_prime_hat:")
        # print(q_hat.shape)
        # print("q_hat:")
        # print(q_hat.min(), q_hat.max(), q_hat.mean(), q_hat.shape)
        # print(pres.min(), pres.max(), pres.mean())
        # q_hat = torch.cumulative_trapezoid(q_hat, 10**pres.squeeze())

        # print("q_hat after trapezoid:")
        # print(q_hat.min(), q_hat.max(), q_hat.mean())

        return out, q_hat


        
