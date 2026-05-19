"""
xyz_to_pt.py
Convert xyz_ln_all/*.xyz → train.pt / val.pt for LigandDiff training.

Key differences from the original xyz_generate_data_train.py:
  - BondedOct=False  : detects any CN (4-10+), not just 6
  - col_idx cap at 6 : truncate to at most 6 ligand groups (model limit)
  - Skips structures where ligand_breakdown finds 0 ligands
  - 80/20 train/val split (or --val_dir for explicit val set)

Usage:
  python xyz_to_pt.py --xyz_dir xyz_ln_all --out_dir data/Ln_data_new
"""

import argparse, os, random, sys, torch
sys.path.insert(0, '/home/scli/diffusion_model/gvp_layer_Ln_LigandDiff')

from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown
from torch_geometric.data import Data

ELEMENT_TO_IDX = {'C': 0, 'N': 1, 'O': 2, 'S': 3,
                  'Br': 4, 'Cl': 5, 'P': 6, 'F': 7}
LN_ELEMENTS = {'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd',
               'Tb','Dy','Ho','Er','Tm','Yb','Lu'}
CHARGES = {
    'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,
    'Br':35,'I':53,'B':5,'Al':13,'Ga':31,'Li':3,'Na':11,
    'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':61,'Eu':62,
    'Gd':63,'Tb':64,'Dy':65,'Ho':66,'Er':67,'Tm':68,'Yb':69,'Lu':70,
    'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,
    'Ru':44,'Pd':46,'Pt':78,
}
MAX_LIG_GROUPS = 6


def extract_xyz(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    coords, one_hot_vecs, charges = [], [], []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        if elem == 'H' or elem == 'D':
            continue
        if elem not in CHARGES:
            continue
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        coords.append([x, y, z])
        charges.append(CHARGES[elem])
        vec = [0] * 8
        if elem in ELEMENT_TO_IDX:
            vec[ELEMENT_TO_IDX[elem]] = 1
        one_hot_vecs.append(vec)
    if not coords:
        return None, None, None, None
    label = lines[1].strip()
    return (torch.tensor(coords, dtype=torch.float),
            torch.tensor(one_hot_vecs, dtype=torch.float),
            label,
            torch.tensor(charges, dtype=torch.int))


def process_xyz(xyz_path):
    """Convert one xyz file to a list of Data objects (one per ligand)."""
    coords_t, one_hot_t, label, charges_t = extract_xyz(xyz_path)
    if coords_t is None:
        return []

    mol = mol3D()
    try:
        mol.readfromxyz(xyz_path)
    except Exception:
        return []

    try:
        # BondedOct=False: detect any CN
        liglist, ligdents, ligcon = ligand_breakdown(
            mol, silent=True, BondedOct=False)
    except Exception:
        return []

    if not liglist:
        return []

    n_atoms = one_hot_t.shape[0]

    # Build ligand_group_matrix (max 6 columns)
    lig_group_mat = torch.zeros((n_atoms, MAX_LIG_GROUPS))
    for col_idx, positions in enumerate(liglist):
        if col_idx >= MAX_LIG_GROUPS:
            break                          # truncate at 6
        idx0 = torch.tensor(positions, dtype=torch.long) - 1
        valid = idx0[(idx0 >= 0) & (idx0 < n_atoms)]
        if len(valid):
            lig_group_mat[valid, col_idx] = 1

    data_list = []
    for lig_num in range(len(ligcon)):
        data = Data(pos=coords_t)
        data.label          = label
        data.nuclear_charges = charges_t
        data.num_atoms      = torch.tensor(n_atoms, dtype=torch.long)
        data.one_hot        = one_hot_t
        data.ligand_group   = lig_group_mat

        context     = torch.ones(n_atoms)
        ligand_diff = torch.zeros(n_atoms)
        for atom_idx in liglist[lig_num]:
            i = atom_idx - 1
            if 0 <= i < n_atoms:
                ligand_diff[i] = 1
                context[i]     = 0
        data.context     = context
        data.ligand_diff = ligand_diff
        data_list.append(data)

    return data_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz_dir',  default='xyz_ln_all')
    p.add_argument('--out_dir',  default='data/Ln_data_new')
    p.add_argument('--val_split',type=float, default=0.1,
                   help='Fraction for validation (default 0.1)')
    p.add_argument('--seed',     type=int,   default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    xyz_files = sorted([f for f in os.listdir(args.xyz_dir)
                        if f.endswith('.xyz')])
    random.shuffle(xyz_files)

    n_val   = max(1, int(len(xyz_files) * args.val_split))
    val_set = set(xyz_files[:n_val])

    train_data, val_data = [], []
    skipped = 0

    for i, fname in enumerate(xyz_files):
        path = os.path.join(args.xyz_dir, fname)
        items = process_xyz(path)
        if not items:
            skipped += 1
            continue
        if fname in val_set:
            val_data.extend(items)
        else:
            train_data.extend(items)

        if (i + 1) % 500 == 0 or i < 3:
            print(f'  [{i+1:5d}/{len(xyz_files)}]  '
                  f'train={len(train_data)}  val={len(val_data)}  '
                  f'skipped={skipped}', flush=True)

    torch.save(train_data, os.path.join(args.out_dir, 'train.pt'))
    torch.save(val_data,   os.path.join(args.out_dir, 'val.pt'))

    print(f'\n=== Done ===')
    print(f'  Train samples : {len(train_data)}')
    print(f'  Val samples   : {len(val_data)}')
    print(f'  Skipped files : {skipped}')
    print(f'  Saved to      : {args.out_dir}/')


if __name__ == '__main__':
    main()
