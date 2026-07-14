"""Literature-algorithm metric for the single-ligand GVP DM (DM-based-on-LigandDiff).

Reproduces how LigandDiff computes V/U/N, using the SAME molSimplify whole-complex ligand split as
the multi-LigandDiff lit-metric (NOT the extract_ligand path, whose OpenBabel connectivity collapses
for coordination ligands -- see the DM repo README metric note). Per generated complex:
  sanitycheck (molSimplify) -> ligand list -> keep the GENERATED ligand (ligand_diff mask) ->
  OpenBabel build_mol -> Chem.SanitizeMol (validity), GetMolFrags==1 (connectivity),
  canonical Chem.MolToSmiles (uniqueness), membership-in-train_smiles (novelty).
RAW geometry, generated ligand only, no save-gate. Run from the gvp repo root in the ldtest env."""
import os, glob, argparse, torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from src import utils
from src.lightning import DDPM
from src.molecule_builder import build_mol, sanitycheck
from generate import read_molecule, reform_data


def run(seeds_dir, model, train_smiles_path, n_samples, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = set(l.strip() for l in open(train_smiles_path) if l.strip())
    seed_files = sorted(glob.glob(os.path.join(seeds_dir, '*.xyz')))
    dataset = []
    for f in seed_files:
        try:
            dataset.extend(read_molecule(f))
        except Exception as e:
            print('skip seed', os.path.basename(f), e, flush=True)
    dataset = dataset * n_samples
    print(f'{len(dataset)} samples over {len(seed_files)} seeds (n_samples={n_samples})', flush=True)
    data = reform_data(dataset, device, ligand_sizes='random')
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dl = DataLoader(data, batch_size=min(batch_size, len(data)), shuffle=False)

    all_mols = []
    with torch.no_grad():
        for b, dat in enumerate(dl):
            pos0 = dat['pos']; seg = dat.batch
            nb = int(seg.max()) + 1
            ld = dat['ligand_diff'].view(-1)
            ctx = dat['context'].view(-1, 1)
            metals = [dat['nuclear_charges'][seg == i][0] for i in range(nb)]
            fmean = scatter_add(pos0 * ctx, seg, dim=0) / scatter_add(ctx, seg, dim=0).view(-1, 1)
            try:
                chain = ddpm.sample_chain(dat, keep_frames=100)
            except utils.FoundNaNException:
                continue
            except Exception as e:
                print('sample_chain err', e, flush=True); continue
            x = chain[0][:, :3] + fmean[seg]
            oh = chain[0][:, 3:]
            for i in torch.unique(seg):
                mask = (seg == i)
                positions = x[mask]
                ats = oh[mask].argmax(dim=1)
                ld_i = ld[mask]
                gen_atoms = set((ld_i == 1).nonzero(as_tuple=True)[0].tolist())
                metal = metals[i]
                try:
                    overlapping, liglist = sanitycheck(positions, ats, metal)
                except Exception:
                    continue
                genligs = [lig for lig in liglist if any(a in gen_atoms for a in lig)]
                for lig in genligs:
                    try:
                        m = build_mol(positions[lig], ats[lig])
                        if m is not None:
                            all_mols.append(m)
                    except Exception:
                        pass
            if b % 20 == 0:
                print(f'  batch {b}: gen_ligands={len(all_mols)}', flush=True)

    valid = []
    for m in all_mols:
        try:
            Chem.SanitizeMol(m); valid.append(m)
        except Exception:
            pass
    connected = [m for m in valid if len(Chem.rdmolops.GetMolFrags(m, asMols=True)) == 1]
    smis = [Chem.MolToSmiles(m) for m in connected]
    uniq = set(smis)
    novel = [s for s in uniq if s not in train]

    def p(a, b): return round(100 * a / max(b, 1), 1)
    print('=== LITMETRICS (LigandDiff algorithm: molSimplify split + OpenBabel build_mol + SanitizeMol, RAW, generated-only) ===', flush=True)
    print(f'gen_ligands={len(all_mols)}')
    print(f'VALIDITY     (valid/gen)       = {len(valid)}/{len(all_mols)} = {p(len(valid), len(all_mols))}%')
    print(f'CONNECTIVITY (connected/valid) = {len(connected)}/{len(valid)} = {p(len(connected), len(valid))}%')
    print(f'UNIQUENESS   (unique/connected)= {len(uniq)}/{len(connected)} = {p(len(uniq), len(connected))}%')
    print(f'NOVELTY      (novel/unique)    = {len(novel)}/{len(uniq)} = {p(len(novel), len(uniq))}%   (ref {len(train)} train SMILES)')
    print('LITMETRICS_DONE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds_dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--train_smiles', required=True)
    ap.add_argument('--n_samples', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=8)
    a = ap.parse_args()
    run(a.seeds_dir, a.model, a.train_smiles, a.n_samples, a.batch_size)
