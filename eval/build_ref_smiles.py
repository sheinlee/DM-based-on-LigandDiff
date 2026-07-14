"""
Build the NOVELTY reference set from the actual TRAINING tensor a model was trained on.

The gvp repo's data/train_smiles.csv is the ORIGINAL transition-metal LigandDiff set and is the
WRONG reference for the Ln-retrained models. So we derive the reference directly from the .pt the
model saw, guaranteeing it matches (Ln-only for Ln models; TM+Ln for the combined models) and that
bond perception is identical to the generated side (fair novelty comparison).

For each UNIQUE complex (dedup by label): split off the metal (atom 0), find ligand connected
components on the covalent-radius graph, and record each ligand as
   (a) a canonical SMILES  (rdDetermineBonds, trying a few net charges; stereo dropped), and
   (b) a Weisfeiler-Lehman hash of its element-labeled connectivity graph  (bond-order-free,
       always computable -> robust for coordination ligands where bond perception is unreliable).

Outputs two files next to --out:  <out>.smi  (one canonical SMILES per line)
                                  <out>.wl   (one WL hash per line)

Run (ldtest env):
  python build_ref_smiles.py --train_pt <train.pt> --out ref_Ln
"""
import argparse, json, signal
import numpy as np
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

IDX2ATOM = {0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'Br', 5: 'Cl', 6: 'P', 7: 'F'}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)   # rdDetermineBonds can hang on big multidentate ligands
COV = {'C':0.76,'N':0.71,'O':0.66,'S':1.05,'Br':1.20,'Cl':1.02,'P':1.07,'F':0.57,
       'H':0.31,'B':0.84,'Si':1.11,'Se':1.20,'As':1.19,'I':1.39}


def ligand_components(elements, coords, tol=1.3):
    coords = np.asarray(coords, float)
    keep = [i for i, e in enumerate(elements) if i != 0 and e != 'H']  # atom 0 = metal
    G = nx.Graph()
    for i in keep:
        G.add_node(i, element=elements[i])
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            i, j = keep[a], keep[b]
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.4 < d < (COV.get(elements[i], 0.77) + COV.get(elements[j], 0.77)) * tol:
                G.add_edge(i, j)
    return [G.subgraph(c).copy() for c in nx.connected_components(G)]


def smiles_of(els, coords):
    """els/coords already sliced to ONE ligand's own atoms. Best-effort, hard-capped."""
    n = len(els)
    if n < 2:
        return None
    rw = Chem.RWMol()
    for e in els:
        rw.AddAtom(Chem.Atom(e))
    conf = Chem.Conformer(n)
    for k in range(n):
        conf.SetAtomPosition(k, Point3D(*[float(x) for x in coords[k]]))
    base = rw.GetMol(); base.AddConformer(conf)
    for chg in (0, -1):                                        # Ln ligands: neutral / anionic mostly
        try:
            signal.setitimer(signal.ITIMER_REAL, 4.0)          # cap pathological bond search
            try:
                m = Chem.Mol(base)
                rdDetermineBonds.DetermineBonds(m, charge=chg)
                Chem.SanitizeMol(m)
                smi = Chem.MolToSmiles(m, isomericSmiles=False)  # graph identity, drop noisy stereo
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)        # ALWAYS cancel during unwind
            return smi
        except BaseException:                                  # _Timeout or any RDKit error
            continue
    return None


def main(a):
    import torch
    v = torch.load(a.train_pt, map_location='cpu')

    # ---- PHASE 1: WL-graph reference (fast, robust; the backbone novelty set) ----
    seen = set(); wl = set(); ligands = []
    for x in v:
        lab = str(x['label'])
        if lab in seen:
            continue
        seen.add(lab)
        try:
            pos = x['pos'].numpy()
            oh = x['one_hot'].argmax(1).tolist()
            els = ['M'] + [IDX2ATOM.get(k, 'C') for k in oh[1:]]
            for comp in ligand_components(els, pos, a.tol):
                wl.add(wl_hash(comp, node_attr='element'))
                idx = sorted(comp.nodes)
                if 2 <= len(idx) <= a.max_atoms:               # stash for SMILES pass
                    ligands.append(([els[i] for i in idx], pos[idx].copy()))
        except BaseException:
            pass
    with open(a.out + '.wl', 'w') as f:
        f.write('\n'.join(sorted(wl)))
    print(f"WL DONE: {len(seen)} unique complexes | {len(wl)} unique WL graphs | "
          f"{len(ligands)} ligands (<= {a.max_atoms} atoms) queued for SMILES", flush=True)

    # ---- PHASE 2: SMILES reference (best-effort, checkpointed) ----
    smi = set(); nok = 0
    for k, (els, co) in enumerate(ligands):
        s = smiles_of(els, co)
        if s:
            smi.add(s); nok += 1
        if (k + 1) % 500 == 0:
            with open(a.out + '.smi', 'w') as f:
                f.write('\n'.join(sorted(smi)))
            print(f"  SMILES {k+1}/{len(ligands)} | {len(smi)} unique | {nok} perceived", flush=True)
    with open(a.out + '.smi', 'w') as f:
        f.write('\n'.join(sorted(smi)))
    print(f"SMILES DONE: {len(smi)} unique ligand SMILES "
          f"({nok}/{len(ligands)} ligands perceived) -> wrote {a.out}.smi", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_pt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tol', type=float, default=1.3)
    p.add_argument('--max_atoms', type=int, default=32, help='skip SMILES for ligands bigger than this')
    main(p.parse_args())
