import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_scatter import scatter_add
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.typing import OptTensor, SparseTensor



class RBFExpansion(nn.Module):
    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale = None,
    ):
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

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (distance - self.centers) ** 2
        )

class EdgeGatedGraphConv(nn.Module):
    
    def __init__(self, input_features: int, output_features: int, residual: bool = True):

        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)
        
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)
    
    def forward(self, node_feats, edge_feats, edge_index):
        idx_i, idx_j = edge_index
        dim_size = node_feats.shape[0]
        e_src = self.src_gate(node_feats)[idx_i]
        e_dst = self.dst_gate(node_feats)[idx_j]

        y = e_src + e_dst + self.edge_gate(edge_feats)
        sigma = torch.sigmoid(y)
        bh = self.dst_update(node_feats)[idx_j]
        m = bh*sigma
        # print("m, idx_i",m.shape, idx_i.shape)
        sum_sigma_h = scatter_add(m, idx_i, 0, dim_size=dim_size)

        sum_sigma = scatter_add(sigma, idx_i, 0, dim_size=dim_size)

        h = sum_sigma_h/(sum_sigma+1e-6)
        # print("nodefeats, h", node_feats.shape, h.shape)
        x = self.src_update(node_feats) + h

        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(y))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y

class ALIGNNConv(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(self, x, y, z, edge_index, edge_index_triplets):

        m, z = self.edge_update(y, z, edge_index_triplets)
        x, y = self.node_update(x, m, edge_index)

        return x, y, z

class MLPLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        return F.silu(self.bn(self.layer(x)))

class ALIGNN(nn.Module):

    def __init__(self, node_input_features=1, embedding_features=64, triplet_input_features=40, hidden=256, out_size=1, mx_d=8, centers=80, a_layers=4, g_layers=4, *args,**kwargs):
        super().__init__()

        self.atom_embedding = nn.Sequential(MLPLayer(node_input_features, hidden))
        self.edge_embedding = nn.Sequential(RBFExpansion(vmin=2.0, vmax=4.0, bins=centers),
                                            MLPLayer(centers, embedding_features),
                                            MLPLayer(embedding_features, hidden))
        self.angle_embedding = nn.Sequential(RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
                                             MLPLayer(triplet_input_features, embedding_features), 
                                             MLPLayer(embedding_features, hidden))

        self.alignn_layers = nn.ModuleList([ALIGNNConv(hidden, hidden) for _ in range(a_layers)])
        self.gcn_layers = nn.ModuleList([EdgeGatedGraphConv(hidden, hidden) for _ in range(g_layers)])

        self.readout = MeanAggregation()
        self.out = nn.Linear(hidden, out_size)


    def create_line_graph(self, edge_index):
        """
        Calculates i,j,k triplets for T-O-T-O-T bonds

        edge_index: (2, M) tensor
        """
        n = edge_index.max().item() + 1

        ind_i, ind_j = edge_index

        value = torch.arange(ind_j.size(0), device=ind_j.device)
        adj_t = SparseTensor(row=ind_i, col=ind_j, value=value,
                                sparse_sizes=(n,n))
        adj_t_row = adj_t[ind_j]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = ind_i.repeat_interleave(num_triplets)
        idx_j = ind_j.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        triplet_index = torch.stack([idx_ji, idx_kj], dim=0)

        return triplet_index
    
    def compute_bond_cosines(self, triplet_index, edge_attr):
        """Compute bond angle cosines from bond displacement vectors."""
        # line graph edge: (a, b), (b, c)
        # `a -> b -> c`
        # use law of cosines to compute angles cosines
        # negate src bond so displacements are like `a <- b -> c`
        # cos(theta) = ba \dot bc / (||ba|| ||bc||)
        u, v = triplet_index
        r1 = -edge_attr[u]
        r2 = edge_attr[v]
        bond_cosine = torch.sum(r1 * r2, dim=1) / (
            torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
        )
        bond_cosine = torch.clamp(bond_cosine, -1, 1)
        return bond_cosine
        

    def forward(self, data):  

        triplet_index = self.create_line_graph(data.edge_index)
        triplet_attr = self.compute_bond_cosines(triplet_index, data.edge_attr).unsqueeze(-1)

        x = self.atom_embedding(data.x)
        y = self.edge_embedding(data.edge_attr)
        z = self.angle_embedding(triplet_attr)

        for layer in self.alignn_layers:
            x, y, z = layer(x, y, z, data.edge_index, triplet_index)

        for layer in self.gcn_layers:
            x, y = layer(x, y, data.edge_index)

        h = self.readout(x, data.batch)
        out = self.out(h)

        return out.squeeze()