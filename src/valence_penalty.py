"""
Soft differentiable valence penalty.

Core idea:
  bond_prob(i,j) = sigmoid((cutoff - dist(i,j)) / temperature)
  soft_valence(i) = Σ_j bond_prob(i,j)
  penalty = mean over atoms of relu(soft_valence - max_valence)²

Gradient flows: penalty → dist → positions → eps_hat → model weights
→ model learns to place atoms so that valence constraints are satisfied.

Only applied to LIGAND atoms (ligand_diff == 1); coordination bonds to
the metal are automatically excluded because the metal is a context atom.
"""

import torch
import torch.nn.functional as F

# [C, N, O, S, Br, Cl, P, F]
_MAX_VALENCE = torch.tensor([4., 3., 2., 6., 1., 1., 5., 1.])

# Distance cutoff for "bonding region" (Å); sigmoid is 0.5 here
BOND_CUTOFF = 2.0

# Sharpness of sigmoid transition (smaller = sharper)
BOND_TEMP = 0.3


def soft_valence_penalty(x_pred, atom_types_onehot, ligand_diff, batch_seg,
                          cutoff=BOND_CUTOFF, temperature=BOND_TEMP):
    """
    x_pred           : [N, 3]  predicted clean positions (all atoms)
    atom_types_onehot: [N, 8]  one-hot (C,N,O,S,Br,Cl,P,F)
    ligand_diff      : [N] or [N,1]  1 = ligand atom, 0 = context
    batch_seg        : [N]  batch index per atom

    Returns scalar penalty (mean over batch elements).
    """
    if ligand_diff.dim() == 2:
        ligand_diff = ligand_diff.squeeze(-1)

    max_val_table = _MAX_VALENCE.to(x_pred.device)   # [8]
    batch_size = int(batch_seg.max().item()) + 1
    total_penalty = x_pred.new_zeros(1)
    n_valid = 0

    for b in range(batch_size):
        mask = (batch_seg == b) & (ligand_diff > 0.5)
        n = mask.sum().item()
        if n < 2:
            continue

        x_lig   = x_pred[mask]               # [n, 3]
        types   = atom_types_onehot[mask]     # [n, 8]
        max_val = (types * max_val_table).sum(-1)  # [n]

        # Pairwise distances [n, n]
        diff = x_lig.unsqueeze(0) - x_lig.unsqueeze(1)   # [n,n,3]
        dist = diff.norm(dim=-1)                           # [n,n]

        # Soft bond probability; zero self-pairs
        eye  = torch.eye(n, device=x_pred.device)
        prob = torch.sigmoid((cutoff - dist) / temperature) * (1 - eye)

        # Soft valence per ligand atom
        soft_val = prob.sum(-1)                           # [n]

        # Penalise over-valent atoms: relu(soft_val - max_val)²
        viol = F.relu(soft_val - max_val)
        total_penalty = total_penalty + (viol ** 2).mean()
        n_valid += 1

    return total_penalty / max(n_valid, 1)
