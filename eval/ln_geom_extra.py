"""Coordination-geometry validation beyond bond length: donor-M-donor ANGLE distribution and
first-shell CN, generated (raw) vs the CSD training reference. Tests whether generated Ln
complexes form sensible coordination polyhedra, not just correct distances."""
import glob, os, argparse
import numpy as np

LAN = "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
LANSET = set(LAN)
# train.pt uses REAL atomic numbers (verified: 57-71 present, Pm=61 NOT skipped)
IDX2M = {57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",
         67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu"}
Zel = {6:"C",7:"N",8:"O",9:"F",15:"P",16:"S",17:"Cl",35:"Br",53:"I"}
HET = {"O","N","F","S","Cl","Br","I"}
CUT = 2.9


def read_xyz(p):
    L = open(p).read().splitlines(); n = int(L[0].split()[0]); els, xyz = [], []
    for ln in L[2:2+n]:
        s = ln.split(); els.append(s[0]); xyz.append([float(v) for v in s[1:4]])
    return els, np.array(xyz, float)


def angles_cn_from_donors(metal_pos, donor_pos):
    cn = len(donor_pos)
    angs = []
    v = donor_pos - metal_pos
    nv = np.linalg.norm(v, axis=1)
    for i in range(cn):
        for j in range(i+1, cn):
            c = np.dot(v[i], v[j]) / (nv[i]*nv[j] + 1e-9)
            angs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return cn, angs


def from_xyz_dir(dirpath):
    cns = []; angs = []
    for f in glob.glob(os.path.join(dirpath, "**", "*.xyz"), recursive=True):
        try: els, xyz = read_xyz(f)
        except Exception: continue
        if els[0] not in LANSET: continue
        d = np.linalg.norm(xyz - xyz[0], axis=1)
        idx = [i for i in range(1, len(els)) if d[i] <= CUT and els[i] in HET]
        if len(idx) < 2: continue
        cn, a = angles_cn_from_donors(xyz[0], xyz[idx])
        cns.append(cn); angs.extend(a)
    return np.array(cns), np.array(angs)


def from_pt(ptpath):
    import torch
    d = torch.load(ptpath, map_location="cpu", weights_only=False)
    cns = []; angs = []; seen = set()
    for it in d:
        lab = getattr(it, "label", None)
        if lab in seen: continue
        seen.add(lab)
        Z = [int(x) for x in it.nuclear_charges.tolist()]; pos = it.pos.numpy()
        mi = [i for i, z in enumerate(Z) if z in IDX2M]
        if len(mi) != 1: continue
        mi = mi[0]; dist = np.linalg.norm(pos - pos[mi], axis=1)
        idx = [i for i in range(len(Z)) if i != mi and dist[i] <= CUT and Zel.get(Z[i]) in HET]
        if len(idx) < 2: continue
        cn, a = angles_cn_from_donors(pos[mi], pos[idx])
        cns.append(cn); angs.extend(a)
    return np.array(cns), np.array(angs)


def hist(a, edges):
    h, _ = np.histogram(a, bins=edges, density=True)
    return h


def wasser1(a, b):
    a = np.sort(a); b = np.sort(b)
    q = np.linspace(0, 1, 200)
    return np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pt", required=True); ap.add_argument("--gen_dir", required=True)
    a = ap.parse_args()
    ecn, eang = from_pt(a.train_pt)
    gcn, gang = from_xyz_dir(a.gen_dir)
    print(f"complexes: exp={len(ecn)}  gen={len(gcn)}")
    print("\n=== first-shell CN distribution (heteroatom donors <=2.9 A) ===")
    print(f"  EXP  CN mean {ecn.mean():.2f} +- {ecn.std():.2f}   median {np.median(ecn):.0f}")
    print(f"  GEN  CN mean {gcn.mean():.2f} +- {gcn.std():.2f}   median {np.median(gcn):.0f}")
    for lo in range(3, 13):
        pe = 100*np.mean(ecn == lo); pg = 100*np.mean(gcn == lo)
        if pe > 1 or pg > 1: print(f"    CN {lo:2d}:  exp {pe:4.1f}%   gen {pg:4.1f}%")
    print("\n=== donor-M-donor ANGLE distribution (degrees) ===")
    print(f"  EXP  n_angles={len(eang)}  mean {eang.mean():.1f} +- {eang.std():.1f}")
    print(f"  GEN  n_angles={len(gang)}  mean {gang.mean():.1f} +- {gang.std():.1f}")
    print(f"  --> Wasserstein-1 distance (gen vs exp) = {wasser1(gang, eang):.1f} deg")
    edges = np.arange(40, 181, 15)
    he, hg = hist(eang, edges), hist(gang, edges)
    print("  angle histogram (density x1000):")
    print("     bin:  " + "  ".join(f"{edges[i]:3.0f}-{edges[i+1]:3.0f}" for i in range(len(edges)-1)))
    print("     EXP:  " + "  ".join(f"{he[i]*1000:6.1f}" for i in range(len(he))))
    print("     GEN:  " + "  ".join(f"{hg[i]*1000:6.1f}" for i in range(len(hg))))
