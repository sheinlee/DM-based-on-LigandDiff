"""
Literature generative-quality metrics for Ln complex generators, on xtb-relaxed outputs.

Reports the standard V/U/N trio (as in LigandDiff / multi-LigandDiff) PLUS our complex-level
chemistry checks, all on the SAME complex set (optional CN 8-10 filter):

  per-ligand VALIDITY  = valid ligands / total ligands
      valid = RDKit perceives bonds from 3D (rdDetermineBonds, few charges) and SanitizeMol OK.
  UNIQUENESS = unique / total among valid ligands            (SMILES-based and WL-graph-based)
  NOVELTY    = (valid-unique NOT in training reference) / valid-unique
      SMILES-based (canonical, stereo-free) AND WL-graph-based (connectivity, bond-order-free).
      The graph form is the robust number for coordination ligands where bond perception is shaky
      -- it mirrors the topology rubric we use for recall.
  complex YIELD (meaningful) = complexes with no clash + every ligand valid + no lone-atom fragment.

Novelty needs --ref built by build_ref_smiles.py (<ref>.smi + <ref>.wl) from the model's own
training tensor. Recall is measured separately by eval_topology_recall.py.

Run (ldtest env):
  python eval_generative_metrics.py --gen_dir <relaxed_dir> --ref ref_Ln \
         [--val_pt <val.pt> --cn_min 8 --cn_max 10]
"""
import os, glob, argparse, signal
import numpy as np
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)   # rdDetermineBonds can hang on big multidentate ligands

COV = {'C':0.76,'N':0.71,'O':0.66,'S':1.05,'Br':1.20,'Cl':1.02,'P':1.07,'F':0.57,
       'H':0.31,'B':0.84,'Si':1.11,'Se':1.20,'As':1.19,'I':1.39}
LANTH = set('La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split())
DBLOCK = set('Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg'.split())
METALS = LANTH | DBLOCK


def read_xyz(path):
    L = open(path).read().splitlines(); n = int(L[0].split()[0])
    els, xyz = [], []
    for ln in L[2:2 + n]:
        p = ln.split(); els.append(p[0]); xyz.append([float(v) for v in p[1:4]])
    return els, np.array(xyz)


def ligand_graphs(elements, coords, tol=1.3):
    """element-labeled covalent graph over non-metal heavy atoms -> list of subgraphs."""
    coords = np.asarray(coords, float)
    keep = [i for i, e in enumerate(elements) if i != 0 and e not in METALS and e != 'H']
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


def ligand_smiles(elements, coords, nodes):
    idx = sorted(nodes)
    rw = Chem.RWMol()
    for i in idx:
        rw.AddAtom(Chem.Atom(elements[i]))
    conf = Chem.Conformer(len(idx))
    for k, i in enumerate(idx):
        conf.SetAtomPosition(k, Point3D(*[float(x) for x in coords[i]]))
    base = rw.GetMol(); base.AddConformer(conf)
    for chg in (0, -1, 1, -2, 2):
        try:
            signal.setitimer(signal.ITIMER_REAL, 6.0)
            try:
                m = Chem.Mol(base)
                rdDetermineBonds.DetermineBonds(m, charge=chg)
                Chem.SanitizeMol(m)
                smi = Chem.MolToSmiles(m, isomericSmiles=False)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            return smi
        except BaseException:
            continue
    return None


def main(a):
    # ---- optional CN filter set ----
    needed = None
    if a.val_pt and (a.cn_min > 0 or a.cn_max < 99):
        import torch
        v = torch.load(a.val_pt, map_location='cpu'); cn = {}
        for x in v:
            lab = str(x['label'])
            if lab in cn:
                continue
            p = x['pos'].numpy(); d = np.linalg.norm(p[1:] - p[0], axis=1)
            cn[lab] = int((d < 2.9).sum())
        needed = set(l for l, c in cn.items() if a.cn_min <= c <= a.cn_max)

    # ---- reference sets for novelty ----
    ref_smi = ref_wl = None
    if a.ref:
        if os.path.exists(a.ref + '.smi'):
            ref_smi = set(l for l in open(a.ref + '.smi').read().splitlines() if l)
        if os.path.exists(a.ref + '.wl'):
            ref_wl = set(l for l in open(a.ref + '.wl').read().splitlines() if l)

    seeds = sorted(d for d in glob.glob(os.path.join(a.gen_dir, '*')) if os.path.isdir(d))
    n_lig = n_valid = 0
    valid_smis = []                 # canonical SMILES of every VALID ligand (with multiplicity)
    valid_wls = []                  # WL hash of every VALID ligand
    n_cplx = n_meaningful = 0
    for sd in seeds:
        lab = os.path.basename(sd); lab = lab[:-4] if lab.endswith('_sub') else lab
        if needed is not None and lab not in needed:
            continue
        for f in glob.glob(os.path.join(sd, '**', '*.xyz'), recursive=True):
            try:
                els, xyz = read_xyz(f)
            except Exception:
                continue
            n_cplx += 1
            # clash check for the complex-level "meaningful" flag
            from scipy.spatial.distance import pdist
            clash = len(xyz) > 1 and pdist(xyz).min() < 0.9
            graphs = ligand_graphs(els, xyz, a.tol)
            cplx_ok = (not clash) and len(graphs) > 0
            for g in graphs:
                nodes = list(g.nodes)
                if len(nodes) < 2:
                    cplx_ok = False
                    continue
                n_lig += 1
                s = ligand_smiles(els, xyz, nodes)
                if s is not None:
                    n_valid += 1
                    valid_smis.append(s)
                    valid_wls.append(wl_hash(g, node_attr='element'))
                else:
                    cplx_ok = False
            if cplx_ok:
                n_meaningful += 1

    uset, wset = set(valid_smis), set(valid_wls)
    def frac(a_, b_): return 100 * a_ / max(b_, 1)
    tag = '' if needed is None else f' [CN {a.cn_min}-{a.cn_max}]'
    print(f"\n=== GENERATIVE METRICS  ({os.path.basename(a.gen_dir)}){tag} ===")
    print(f"  complexes evaluated:        {n_cplx}")
    print(f"  total ligands:              {n_lig}")
    print(f"  VALIDITY   (valid/ligands): {n_valid}/{n_lig} = {frac(n_valid, n_lig):.1f}%")
    print(f"  UNIQUENESS SMILES (uniq/valid): {len(uset)}/{n_valid} = {frac(len(uset), n_valid):.1f}%")
    print(f"  UNIQUENESS graph  (uniq/valid): {len(wset)}/{n_valid} = {frac(len(wset), n_valid):.1f}%")
    if ref_smi is not None:
        nov = [s for s in uset if s not in ref_smi]
        print(f"  NOVELTY    SMILES (new/uniq):   {len(nov)}/{len(uset)} = {frac(len(nov), len(uset)):.1f}%   (ref {len(ref_smi)} SMILES)")
    if ref_wl is not None:
        novg = [s for s in wset if s not in ref_wl]
        print(f"  NOVELTY    graph  (new/uniq):   {len(novg)}/{len(wset)} = {frac(len(novg), len(wset)):.1f}%   (ref {len(ref_wl)} graphs)")
    print(f"  complex YIELD (meaningful): {n_meaningful}/{n_cplx} = {frac(n_meaningful, n_cplx):.1f}%")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gen_dir', required=True)
    p.add_argument('--ref', default=None, help='prefix built by build_ref_smiles.py (<ref>.smi/.wl)')
    p.add_argument('--val_pt', default=None)
    p.add_argument('--tol', type=float, default=1.3)
    p.add_argument('--cn_min', type=int, default=0)
    p.add_argument('--cn_max', type=int, default=99)
    main(p.parse_args())
