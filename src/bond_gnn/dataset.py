"""
Build bond training dataset from CIF files + xyz_ln_all.

For each structure:
  - Read atom positions from xyz file
  - Parse _geom_bond_* from the corresponding CIF file
  - Extract LIGAND-ONLY bonds (skip metal-ligand coordination bonds)
  - Label all pairs within BOND_CUTOFF as bonded (1) or not (0)
"""

import os, re, math, sys
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.bond_gnn.model import ATOM2IDX, N_ATOM_TYPES, BOND_CUTOFF

LN_SYMS = {'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu'}


def _parse_xyz_ligand_only(xyz_path):
    """Return (positions, atom_symbols) excluding the metal atom (line 0)."""
    with open(xyz_path) as f:
        lines = f.readlines()
    atoms, positions = [], []
    for line in lines[3:]:   # skip n_atoms, label, metal
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        if sym in LN_SYMS or sym not in ATOM2IDX:
            continue
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(sym)
        positions.append([x, y, z])
    return torch.tensor(positions, dtype=torch.float), atoms


def _parse_cif_bonds(cif_path):
    """
    Return set of frozensets {label_a, label_b} for LIGAND-LIGAND bonds
    (excluding any bond involving a metal atom label).
    """
    with open(cif_path, errors='replace') as f:
        text = f.read()

    # Find _geom_bond block
    block_m = re.search(
        r'loop_\s*((?:_geom_bond_\S+\s*)+)((?:(?!loop_|data_)[\s\S])*)',
        text, re.M)
    if not block_m:
        return set()

    headers = re.findall(r'_geom_bond_\S+', block_m.group(1))
    col = {h.lower(): i for i, h in enumerate(headers)}
    label1_col = col.get('_geom_bond_atom_site_label_1')
    label2_col = col.get('_geom_bond_atom_site_label_2')
    if label1_col is None or label2_col is None:
        return set()

    tokens = block_m.group(2).split()
    n_cols = len(headers)
    bonds = set()
    i = 0
    while i + n_cols <= len(tokens):
        row = tokens[i:i + n_cols]
        i += n_cols
        if row[0].startswith('_') or row[0].startswith('#'):
            i -= n_cols - 1; continue
        a1, a2 = row[label1_col], row[label2_col]
        # Skip metal-involving bonds
        a1_elem = re.match(r'([A-Z][a-z]?)', a1)
        a2_elem = re.match(r'([A-Z][a-z]?)', a2)
        if not a1_elem or not a2_elem:
            continue
        if a1_elem.group(1) in LN_SYMS or a2_elem.group(1) in LN_SYMS:
            continue
        bonds.add(frozenset([a1, a2]))
    return bonds


def _parse_cif_atom_labels(cif_path):
    """
    Return ordered list of atom labels matching our xyz_to_pt order
    (non-metal heavy atoms from _atom_site loop, H excluded).
    """
    with open(cif_path, errors='replace') as f:
        text = f.read()

    block_m = re.search(
        r'loop_\s*((?:_atom_site_\S+\s*)+)((?:(?!loop_|data_)[\s\S])*)',
        text, re.M)
    if not block_m:
        return []

    headers = re.findall(r'_atom_site_\S+', block_m.group(1))
    col = {h.lower(): i for i, h in enumerate(headers)}
    label_col = col.get('_atom_site_label')
    elem_col  = col.get('_atom_site_type_symbol', col.get('_atom_site_label'))
    if label_col is None:
        return []

    tokens = block_m.group(2).split()
    n_cols = len(headers)
    labels = []
    i = 0
    while i + n_cols <= len(tokens):
        row = tokens[i:i + n_cols]
        i += n_cols
        if row[0].startswith('_') or row[0].startswith('#'):
            i -= n_cols - 1; continue
        label = row[label_col]
        elem_raw = row[elem_col] if elem_col is not None else label
        elem_m = re.match(r'([A-Z][a-z]?)', elem_raw)
        if not elem_m:
            continue
        elem = elem_m.group(1)
        if elem in ('H', 'D') or elem in LN_SYMS:
            continue
        if elem not in ATOM2IDX:
            continue
        labels.append(label)
    return labels


class BondDataset(Dataset):
    """
    Each item: (pos [N,3], one_hot [N,8], bond_labels [E], edge_pairs [E,2])
    where E = all pairs within BOND_CUTOFF.
    """

    def __init__(self, xyz_dir, cif_dir, max_samples=None):
        self.samples = []
        xyz_files = sorted([f for f in os.listdir(xyz_dir) if f.endswith('.xyz')])
        if max_samples:
            xyz_files = xyz_files[:max_samples]

        ok = skip_no_cif = skip_no_bonds = skip_empty = 0

        for xyz_fname in xyz_files:
            cif_code = xyz_fname.replace('_sub.xyz', '')
            xyz_path = os.path.join(xyz_dir, xyz_fname)

            # Find CIF file
            cif_path = self._find_cif(cif_dir, cif_code)
            if cif_path is None:
                skip_no_cif += 1
                continue

            pos, syms = _parse_xyz_ligand_only(xyz_path)
            if pos.size(0) < 2:
                skip_empty += 1
                continue

            # Get bond labels from CIF
            cif_bonds = _parse_cif_bonds(cif_path)
            cif_labels = _parse_cif_atom_labels(cif_path)

            if not cif_bonds or not cif_labels:
                skip_no_bonds += 1
                continue

            # Map CIF labels to our atom index order
            bonded_pairs = self._resolve_bonds(cif_bonds, cif_labels, syms)

            # Build one-hot
            n = len(syms)
            one_hot = torch.zeros(n, N_ATOM_TYPES)
            for i, s in enumerate(syms):
                if s in ATOM2IDX:
                    one_hot[i, ATOM2IDX[s]] = 1

            # Build all pairs within cutoff + labels
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist2 = (diff ** 2).sum(-1)
            mask = (dist2 < BOND_CUTOFF ** 2) & (dist2 > 1e-8)
            src, dst = mask.nonzero(as_tuple=True)

            labels = torch.zeros(src.size(0), dtype=torch.float)
            bonded_set = {(min(i,j), max(i,j)) for i, j in bonded_pairs}
            for k, (i, j) in enumerate(zip(src.tolist(), dst.tolist())):
                if (min(i,j), max(i,j)) in bonded_set:
                    labels[k] = 1.0

            edge_pairs = torch.stack([src, dst], dim=1)
            self.samples.append((pos, one_hot, labels, edge_pairs))
            ok += 1

        print(f'BondDataset: {ok} ok, {skip_no_cif} no-CIF, '
              f'{skip_no_bonds} no-bonds, {skip_empty} empty')

    def _find_cif(self, cif_dir, code):
        for elem_dir in os.listdir(cif_dir):
            p = os.path.join(cif_dir, elem_dir, f'{code}.cif')
            if os.path.exists(p):
                return p
        return None

    def _resolve_bonds(self, cif_bonds, cif_labels, xyz_syms):
        """Map CIF bond label pairs to (idx_i, idx_j) in our xyz ordering."""
        bonds = []
        for pair in cif_bonds:
            pair_list = list(pair)
            if len(pair_list) != 2:
                continue
            a, b = pair_list
            # Strip site symmetry suffix (e.g. 'C1_a' → 'C1')
            a = a.split('_')[0]
            b = b.split('_')[0]
            # Find indices in cif_labels that match
            ai = [i for i, l in enumerate(cif_labels) if l.split('_')[0] == a]
            bi = [i for i, l in enumerate(cif_labels) if l.split('_')[0] == b]
            for i in ai:
                for j in bi:
                    if i < len(xyz_syms) and j < len(xyz_syms):
                        bonds.append((i, j))
        return bonds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Collate variable-size graphs into a single batch."""
    all_pos, all_oh, all_labels, all_edges = [], [], [], []
    offset = 0
    for pos, oh, labels, edges in batch:
        n = pos.size(0)
        all_pos.append(pos)
        all_oh.append(oh)
        all_labels.append(labels)
        all_edges.append(edges + offset)
        offset += n
    return (torch.cat(all_pos), torch.cat(all_oh),
            torch.cat(all_labels), torch.cat(all_edges))
