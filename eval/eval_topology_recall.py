"""
Held-out TOPOLOGY recall for Ln coordination-complex generators.

Rubric (per user's decisions, see memory ln-generation-eval-definition):
  * recall success = the generated complex's FULL ligand set matches the TARGET complex's
    full ligand set as a MULTISET of ligand topologies (every ligand present, each matching).
    For multi (>1 generated ligand) ALL generated ligands must be correct.
  * topology = heavy-atom CONNECTIVITY graph (element-labeled), bond ORDERS/charges ignored
    (unreliable for coordination complexes). Metal atom excluded. Geometry is NOT required to
    match (bond lengths are relaxable) -> we compare graphs, not coordinates.
  * a seed counts as recalled if ANY of its generated samples reproduces the target multiset.

Topology key = Weisfeiler-Lehman hash of the element-labeled covalent-bond graph of each ligand.

Run (from the gvp repo dir, env ldtest):
  python eval_topology_recall.py --val_pt <Ln_data_new/val.pt> --gen_dir <heldout_gen*> [--tol 1.3]
Compare models by pointing --gen_dir at heldout_gen2 (baseline) / heldout_gen_B (B) / ... .
"""
import os, sys, glob, argparse
from collections import Counter
import numpy as np
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash

# Cordero (2008) covalent radii, Angstrom — the 8 ligand elements + common extras
COV = {'C':0.76,'N':0.71,'O':0.66,'S':1.05,'Br':1.20,'Cl':1.02,'P':1.07,'F':0.57,
       'H':0.31,'B':0.84,'Si':1.11,'Se':1.20,'As':1.19,'I':1.39}
# metals to EXCLUDE from ligand graphs (lanthanides + common d-block); atom 0 is also always the metal
LANTH = set('La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split())
DBLOCK = set('Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg'.split())
METALS = LANTH | DBLOCK


def ligand_multiset(elements, coords, tol=1.3, drop_H=True):
    """Element-labeled covalent graph over non-metal atoms -> Counter of per-ligand WL hashes."""
    coords = np.asarray(coords, float)
    keep = [i for i, e in enumerate(elements)
            if i != 0 and e not in METALS and not (drop_H and e == 'H')]  # index 0 == metal
    G = nx.Graph()
    for i in keep:
        G.add_node(i, element=elements[i])
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            i, j = keep[a], keep[b]
            d = float(np.linalg.norm(coords[i] - coords[j]))
            ri, rj = COV.get(elements[i], 0.77), COV.get(elements[j], 0.77)
            if 0.4 < d < (ri + rj) * tol:            # 0.4 guards against duplicate/overlap atoms
                G.add_edge(i, j)
    ligs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    return Counter(wl_hash(g, node_attr='element') for g in ligs)


def read_xyz(path):
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    els, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        els.append(p[0]); xyz.append([float(v) for v in p[1:4]])
    return els, np.array(xyz)


def target_topologies(val_pt, tol, needed=None):
    import torch
    from src import const
    idx2atom = const.IDX2ATOM
    v = torch.load(val_pt, map_location='cpu')
    out, cn = {}, {}
    for x in v:
        lab = str(x['label'])
        if needed is not None and lab not in needed:    # only decompose the targets we need
            continue
        if lab in out:                                  # each complex repeats (masked samples); do once
            continue
        pos = x['pos'].numpy()
        oh = x['one_hot'].argmax(1).tolist()
        els = ['M'] + [idx2atom[k] for k in oh[1:]]     # atom 0 = metal (excluded); rest via IDX2ATOM
        out[lab] = ligand_multiset(els, pos, tol)
        d = np.linalg.norm(pos[1:] - pos[0], axis=1)    # CN = donor atoms within Ln-coord distance
        cn[lab] = int((d < 2.9).sum())
    return out, cn


def partial_overlap(gen, tgt):
    """size of multiset-intersection / target size (fraction of target ligands reproduced)."""
    inter = sum((gen & tgt).values())
    return inter / max(sum(tgt.values()), 1)


def main(a):
    seeds = sorted(d for d in glob.glob(os.path.join(a.gen_dir, '*')) if os.path.isdir(d))
    def lab_of(nm):
        return nm[:-4] if nm.endswith('_sub') else nm
    needed = set(lab_of(os.path.basename(sd)) for sd in seeds)
    tgt, cn = target_topologies(a.val_pt, a.tol, needed)   # only decompose the targets we need
    n_seed = n_have_tgt = n_recall = n_gen0 = n_cnfilt = 0
    best_partials = []
    for sd in seeds:
        name = os.path.basename(sd)
        label = lab_of(name)
        if label not in tgt:
            continue
        if not (a.cn_min <= cn.get(label, -1) <= a.cn_max):   # CN sweet-spot filter (Ln: 8-10)
            n_cnfilt += 1
            continue
        n_seed += 1; n_have_tgt += 1
        t = tgt[label]
        xyzs = glob.glob(os.path.join(sd, '**', '*.xyz'), recursive=True)  # recursive: handles multi's noH/ subdir + gvp flat
        if not xyzs:
            n_gen0 += 1; best_partials.append(0.0); continue
        hit = False; best = 0.0
        for f in xyzs:
            try:
                els, xyz = read_xyz(f)
                g = ligand_multiset(els, xyz, a.tol)
            except Exception:
                continue
            best = max(best, partial_overlap(g, t))
            if g == t:
                hit = True
        best_partials.append(best)
        if hit:
            n_recall += 1
    print(f"\n=== TOPOLOGY RECALL  ({os.path.basename(a.gen_dir)})  [CN filter {a.cn_min}-{a.cn_max}] ===")
    print(f"  seeds excluded by CN filter: {n_cnfilt}")
    print(f"  seeds with a target:        {n_have_tgt}")
    print(f"  seeds with 0 generated xyz: {n_gen0}")
    print(f"  RECALL (full-complex, all ligands): {n_recall}/{n_have_tgt} = {100*n_recall/max(n_have_tgt,1):.1f}%")
    print(f"  mean best partial-overlap (frac of target ligands reproduced): {np.mean(best_partials):.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--val_pt', required=True, help='target source, e.g. ~/diffusion_model_new/data/Ln_data_new/val.pt')
    p.add_argument('--gen_dir', required=True, help='dir of <seed>/<b_i>.xyz generated complexes')
    p.add_argument('--tol', type=float, default=1.3, help='covalent-bond distance tolerance')
    p.add_argument('--cn_min', type=int, default=0, help='min coordination number of target (Ln sweet spot: 8)')
    p.add_argument('--cn_max', type=int, default=99, help='max coordination number of target (Ln sweet spot: 10)')
    main(p.parse_args())
