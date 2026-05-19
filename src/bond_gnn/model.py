"""
Bond GNN: predict ligand atom bonds from 3D positions + atom types.
Replaces openbabel's distance-only ConnectTheDots for ligand atoms.

Architecture: 3-layer MPNN → per-pair bond probability.
Training labels: crystallographic bonds from CIF _geom_bond_* section.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


ATOM_TYPES  = ['C', 'N', 'O', 'S', 'Br', 'Cl', 'P', 'F']
N_ATOM_TYPES = len(ATOM_TYPES)
ATOM2IDX    = {a: i for i, a in enumerate(ATOM_TYPES)}

# Max valence per element (for post-processing)
MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'S': 6, 'F': 1, 'Cl': 1, 'Br': 1, 'P': 5}

# Cutoff for candidate bond pairs (Å)
BOND_CUTOFF = 4.0


class MPLayer(nn.Module):
    """One message-passing step using distance + neighbor types."""

    def __init__(self, hidden_dim):
        super().__init__()
        # Edge MLP: (h_i, h_j, dist) → message
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Node update: (h_i, agg_message) → h_i'
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, dist):
        """
        h         : [N, hidden_dim]
        edge_index: [2, E]  (src, dst)
        dist      : [E, 1]
        """
        src, dst = edge_index
        msg_input = torch.cat([h[src], h[dst], dist], dim=-1)
        msg = self.edge_mlp(msg_input)                        # [E, H]
        agg = scatter_add(msg, dst, dim=0, dim_size=h.size(0)) # [N, H]
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)                           # residual


class BondGNN(nn.Module):
    def __init__(self, hidden_dim=64, n_layers=3, cutoff=BOND_CUTOFF):
        super().__init__()
        self.cutoff = cutoff
        self.node_init = nn.Linear(N_ATOM_TYPES, hidden_dim)
        self.mp_layers = nn.ModuleList(
            [MPLayer(hidden_dim) for _ in range(n_layers)]
        )
        # Bond head: (h_i, h_j, dist) → bond logit
        self.bond_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_edges(self, pos):
        """Return edge_index and distances for all pairs within cutoff."""
        n = pos.size(0)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)        # [N, N, 3]
        dist2 = (diff ** 2).sum(-1)                       # [N, N]
        mask = (dist2 < self.cutoff ** 2) & (dist2 > 1e-8)
        src, dst = mask.nonzero(as_tuple=True)
        dist = dist2[src, dst].unsqueeze(-1).sqrt()       # [E, 1]
        return torch.stack([src, dst]), dist

    def forward(self, pos, atom_types_onehot):
        """
        pos              : [N, 3]
        atom_types_onehot: [N, N_ATOM_TYPES]
        Returns logits   : [E] and edge_index [2, E]
        """
        edge_index, dist = self._build_edges(pos)
        h = self.node_init(atom_types_onehot)
        for layer in self.mp_layers:
            h = layer(h, edge_index, dist)

        src, dst = edge_index
        pair_feat = torch.cat([h[src], h[dst], dist], dim=-1)
        logits = self.bond_head(pair_feat).squeeze(-1)    # [E]
        return logits, edge_index, dist.squeeze(-1)

    @torch.no_grad()
    def predict_bonds(self, pos, atom_types_onehot, threshold=0.5):
        """
        Returns list of (i, j) bonded pairs after valence enforcement.
        atom_types_onehot: [N, 8] or list of atom symbols
        """
        self.eval()
        logits, edge_index, _ = self.forward(pos, atom_types_onehot)
        probs = torch.sigmoid(logits)

        src, dst = edge_index
        # Keep only i < j to avoid duplicates, then sort by prob desc
        keep = src < dst
        src_k, dst_k, prob_k = src[keep], dst[keep], probs[keep]
        order = prob_k.argsort(descending=True)
        src_k, dst_k, prob_k = src_k[order], dst_k[order], prob_k[order]

        bonds = []
        valence = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        for i, j, p in zip(src_k.tolist(), dst_k.tolist(), prob_k.tolist()):
            if p < threshold:
                break
            # Check valence constraint
            sym_i = ATOM_TYPES[atom_types_onehot[i].argmax().item()]
            sym_j = ATOM_TYPES[atom_types_onehot[j].argmax().item()]
            max_i = MAX_VALENCE.get(sym_i, 99)
            max_j = MAX_VALENCE.get(sym_j, 99)
            if valence[i] < max_i and valence[j] < max_j:
                bonds.append((i, j))
                valence[i] += 1
                valence[j] += 1

        return bonds
