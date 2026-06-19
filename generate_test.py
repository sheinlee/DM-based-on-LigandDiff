"""
generate_test.py — Ln complex ligand generation with:
  1. BondedOct=False in parse_complex (supports CN 4-12)
  2. Bond GNN for chemically valid bond prediction
  3. Tanimoto similarity comparison against CSD training set
  4. Relaxed connectivity_thresh and atom_tol for evaluation
"""
import argparse
import os
import numpy as np
import tempfile
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from src import const
from src import utils
from src.lightning import DDPM
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sampling import reform_data
from torch_scatter import scatter_add
from src.molecule_builder import (BasicLigandMetrics, build_mol,
    build_mol_with_bond_gnn, extract_ligand, sanitycheck, write_xyz_file)
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown


def load_train_smiles(path=None, lookup_path=None):
    """
    Load training set SMILES + optional CSD lookup for Task 1 comparison.
    Returns (fps_list, csd_lookup_dict).
      fps_list   : [(smiles, fingerprint), ...]
      csd_lookup : {canonical_smiles: [CSD_code, ...]}  (empty if no lookup file)
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'train_smiles_bondgnn.csv')
    if lookup_path is None:
        lookup_path = os.path.join(os.path.dirname(__file__), 'data', 'smiles_to_csd.json')

    fps = []
    if os.path.exists(path):
        with open(path) as f:
            smiles = [s.strip() for s in f if s.strip()]
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append((smi, AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)))

    csd_lookup = {}
    if os.path.exists(lookup_path):
        import json
        with open(lookup_path) as f:
            csd_lookup = json.load(f)

    return fps, csd_lookup


def tanimoto_analysis(smi, train_fps, csd_lookup=None):
    """
    Compare generated SMILES against training set (CSD).
    Returns (best_tanimoto, best_match_smi, is_exact_match, csd_codes).
      csd_codes: list of CSD refcodes if exact/near match found, else []
    """
    if not train_fps or not smi:
        return 0.0, None, False, []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0, None, False, []
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    best_sim, best_smi = 0.0, None
    for ref_smi, ref_fp in train_fps:
        sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
        if sim > best_sim:
            best_sim, best_smi = sim, ref_smi
    is_exact = (best_sim >= 0.999)
    # Look up CSD codes for the best match
    csd_codes = []
    if best_smi and csd_lookup:
        csd_codes = csd_lookup.get(best_smi, [])
    return best_sim, best_smi, is_exact, csd_codes

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--complex', type=str)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--ligand_sizes', type=str, default='random')
parser.add_argument('--connectivity_thresh', type=float, default=0.8,
                    help='Min fraction of atoms in largest fragment (default 0.8)')
parser.add_argument('--atom_tol', type=int, default=2,
                    help='Tolerance for total_atoms vs natoms check (default 2)')

atom2idx = const.ATOM2IDX
idx2atom = const.IDX2ATOM
charges = const.CHARGES
num_atoms_type = const.NUMBER_OF_ATOM_TYPES
metal_list = const.metals


def reform_pos(xyz_file):
    metal_index = None
    with open(xyz_file, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(tuple(metal_list)):
            metal_index = i
            break
    if metal_index is not None:
        lines.insert(2, lines.pop(metal_index))
        with open(f'{xyz_file[:-4]}_re.xyz', 'w') as file:
            file.writelines(lines)


def parse_complex(filename):
    """Parse input complex xyz; BondedOct=False to handle all coordination numbers."""
    label = filename[:-4]
    data_list = []
    ele = []
    pos = []
    nuclear_charges = []
    H_list = []
    noH_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i in lines[3:]:
        if i.split()[0] == 'H':
            H_list.append(i)
        else:
            noH_list.append(i)
            ele.append(atom2idx[i.split()[0]])
            nuclear_charges.append(charges[i.split()[0]])
            pos.append([float(j) for j in i.split()[1:]])
    noH_list.insert(0, lines[2])
    pos.insert(0, [float(j) for j in lines[2].split()[1:]])
    nuclear_charges.insert(0, charges[lines[2].split()[0]])
    one_hot = torch.zeros(len(ele), 8)
    one_hot[range(len(ele)), ele] = 1
    one_hot = torch.cat([torch.zeros(8).view(1, -1), one_hot], dim=0)
    num_atoms = len(pos)
    pos = torch.tensor(pos)
    nuclear_charges = torch.tensor(nuclear_charges)

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        with open(f'{tmp_file}.xyz', 'w') as file:
            file.write(f"{num_atoms}\n\n")
            for sublist in noH_list:
                file.write(f"{sublist}")
    mol = mol3D()
    mol.readfromxyz(f'{tmp_file}.xyz')
    # BondedOct=False: supports any CN (4-12), matches new training data (xyz_to_pt.py)
    liglist, ligdents, ligcon = ligand_breakdown(mol, silent=True, BondedOct=False)
    print(f'  [parse] Found {len(liglist)} ligands, denticity: {ligdents}')

    f_group = torch.zeros(num_atoms)
    for i in range(len(liglist)):
        f_group[liglist[i]] = i + 1

    ligand_group = torch.zeros((num_atoms, 7))
    ligand_group[range(len(f_group.long())), f_group.long()] = 1

    for k in range(len(liglist)):
        ligand = torch.zeros(num_atoms)
        for i in liglist[k]:
            ligand[i] = 1
        context = 1 - ligand
        data = Data(pos=pos, label=label, context=context, nuclear_charges=nuclear_charges,
                    ligand_diff=ligand, num_atoms=num_atoms, one_hot=one_hot,
                    ligand_group=ligand_group[:, 1:])
        data_list.append(data)
    return data_list


def read_molecule(filename):
    if not filename.endswith('.xyz'):
        raise Exception('Unknown file extension, only .xyz files are supported')
    with open(filename, 'r') as file:
        metal = file.readlines()[2]
        if metal.split()[0] not in metal_list:
            reform_pos(filename)
            print(f'Metal not at line 3; rearranged to {filename[:-4]}_re.xyz')
            return parse_complex(f'{filename[:-4]}_re.xyz')
        else:
            return parse_complex(filename)


def main(outdir, model, complex, batch_size=16, n_samples=1,
         ligand_sizes='random', connectivity_thresh=0.8, atom_tol=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    ddpm.ligand_metrics = BasicLigandMetrics(connectivity_thresh=connectivity_thresh)

    # Load training set SMILES + CSD lookup for topology comparison
    train_fps, csd_lookup = load_train_smiles()
    print(f'Loaded {len(train_fps)} training SMILES for topology comparison'
          f' (CSD lookup: {len(csd_lookup)} entries)')

    dataset = read_molecule(complex) * n_samples
    print(f'{len(dataset)} samples will be generated (n_ligands x n_samples)')
    data = reform_data(dataset, device, ligand_sizes=ligand_sizes)
    batch_size = min(batch_size, len(dataset))
    os.makedirs(outdir, exist_ok=True)

    num_saved = 0
    stats = {'total': 0, 'valid': 0, 'connected': 0, 'no_overlap': 0, 'atom_match': 0}

    with torch.no_grad():
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        for b, data in enumerate(dataloader):
            pos_original = data['pos']
            batch_seg = data.batch
            bs = torch.max(batch_seg) + 1
            ligand_diff = data['ligand_diff'].view(-1, 1)
            context = data['context'].view(-1, 1)
            metals = [data['nuclear_charges'][batch_seg == i][0] for i in range(bs)]
            fixed_mean = scatter_add(pos_original * context, batch_seg, dim=0) / \
                         scatter_add(context, batch_seg, dim=0).view(-1, 1)
            natoms = data['num_atoms']

            try:
                chain_batch = ddpm.sample_chain(data, keep_frames=100)
            except utils.FoundNaNException:
                print(f'  batch {b}: NaN detected, skipping')
                continue

            x_raw = chain_batch[0][:, :3]
            one_hot = chain_batch[0][:, 3:]
            assert one_hot.shape[1] == ddpm.in_node_nf

            x = x_raw + fixed_mean[batch_seg]

            if b == 0:
                lig_mask = ligand_diff[:, 0].bool()
                lig_pos = x[lig_mask]
                dists = torch.cdist(lig_pos[:20], lig_pos[:20])
                print(f'  [debug] ligand pos range: [{lig_pos.min():.2f}, {lig_pos.max():.2f}]')
                print(f'  [debug] pairwise dist min={dists[dists>0].min():.3f} max={dists.max():.3f} mean={dists[dists>0].mean():.3f}')

            ligands = extract_ligand(x, one_hot, ligand_diff, batch_seg)
            # Use Bond GNN for chemically valid bond prediction
            rdmols = [build_mol_with_bond_gnn(pos, types)
                      for pos, types in ligands]
            (validity, connectivity), (valid, connected_mol, connected_index) = \
                ddpm.ligand_metrics.evaluate_rdmols(rdmols)

            stats['total'] += bs.item()
            stats['valid'] += validity
            stats['connected'] += connectivity

            # Tanimoto topology comparison
            for mol in connected_mol:
                try:
                    smi = Chem.MolToSmiles(mol)
                    sim, match_smi, exact, csd_codes = tanimoto_analysis(
                        smi, train_fps, csd_lookup)
                    stats.setdefault('tanimoto_sum', 0.0)
                    stats.setdefault('tanimoto_n', 0)
                    stats.setdefault('exact_matches', 0)
                    stats.setdefault('high_sim', 0)
                    stats['tanimoto_sum'] += sim
                    stats['tanimoto_n'] += 1
                    if exact:
                        stats['exact_matches'] += 1
                    if sim > 0.7:
                        stats['high_sim'] += 1

                    if exact and csd_codes:
                        print(f'    ★ CSD MATCH (Tanimoto=1.0): {smi[:50]}')
                        print(f'      CSD refcodes: {", ".join(csd_codes[:5])}')
                    elif sim > 0.7:
                        csd_info = f' → CSD: {",".join(csd_codes[:3])}' if csd_codes else ''
                        print(f'    ◆ same scaffold (Tanimoto={sim:.3f}): {smi[:50]}{csd_info}')
                    else:
                        print(f'    · novel (Tanimoto={sim:.3f}): {smi[:60]}')
                except Exception:
                    pass

            print(f'  batch {b}: valid={validity}/{bs.item()}, connected={connectivity}/{bs.item()}')

            if connectivity == 0:
                continue

            for i in connected_index:
                positions = x[batch_seg == i]
                atom_types = one_hot[batch_seg == i].argmax(dim=1)
                metal = metals[i]

                overlapping, liglist = sanitycheck(positions, atom_types, metal)
                total_atoms = sum(len(lig) for lig in liglist) + 1

                if not overlapping:
                    stats['no_overlap'] += 1
                # FIX 3: relaxed atom count check (tolerance ±atom_tol)
                atom_match = abs(total_atoms - natoms[i].item()) <= atom_tol
                if atom_match:
                    stats['atom_match'] += 1

                if not overlapping and atom_match:
                    num_saved += 1
                    write_xyz_file(positions, atom_types, f'{outdir}/{b}_{i}', metal)
                else:
                    reason = []
                    if overlapping:
                        reason.append('overlap')
                    if not atom_match:
                        reason.append(f'atom_count {total_atoms} vs {natoms[i].item()}')
                    print(f'    sample {b}_{i} rejected: {", ".join(reason)}')

    # Tanimoto summary
    n_tan = stats.get('tanimoto_n', 0)
    avg_tan = stats.get('tanimoto_sum', 0) / max(n_tan, 1)
    exact   = stats.get('exact_matches', 0)
    high    = stats.get('high_sim', 0)

    print('\n=== Summary ===')
    print(f'  Total generated:   {stats["total"]}')
    print(f'  Valid ligands:     {stats["valid"]}')
    print(f'  Connected (≥{connectivity_thresh:.0%}): {stats["connected"]}')
    print(f'  No overlap:        {stats["no_overlap"]}')
    print(f'  Atom count match:  {stats["atom_match"]}')
    print(f'  Saved complexes:   {num_saved}')
    if n_tan > 0:
        print(f'\n=== Topology vs CSD Training Set ({n_tan} connected ligands) ===')
        print(f'  Avg Tanimoto:      {avg_tan:.3f}')
        print(f'  High sim (>0.7):   {high}/{n_tan}  ← same scaffold as CSD structure')
        print(f'  Exact match (★):   {exact}/{n_tan}  ← topology identical to CSD ligand')
        print(f'  Novel (<0.7):      {n_tan-high}/{n_tan}  ← new chemical scaffold')
    print(f'\nOutput directory: {outdir}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.outdir, args.model, args.complex, args.batch_size,
         args.n_samples, args.ligand_sizes, args.connectivity_thresh, args.atom_tol)
