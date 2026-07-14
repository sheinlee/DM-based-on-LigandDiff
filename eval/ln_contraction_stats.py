"""Rigorous lanthanide-contraction stats on GENERATED donors (no regeneration).
Adds: Shannon-ionic-radius x-axis, slope, La->Lu endpoint drop, bootstrap 95% CI on r,
and a PERMUTATION control (shuffle metal labels -> null r distribution -> p-value). The
permutation test is the key control: if the observed metal-dependent trend is only a measurement
artifact (metal-agnostic), shuffling labels reproduces r; if the MODEL genuinely places donors at
metal-appropriate distances, the observed r sits far outside the shuffled null."""
import glob, os, argparse, random
import numpy as np

LAN = "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
LANSET = set(LAN)
# Shannon effective ionic radii of Ln(3+), CN=6 (Angstrom)
RADIUS = {"La":1.032,"Ce":1.01,"Pr":0.99,"Nd":0.983,"Pm":0.97,"Sm":0.958,"Eu":0.947,"Gd":0.938,
          "Tb":0.923,"Dy":0.912,"Ho":0.901,"Er":0.890,"Tm":0.880,"Yb":0.868,"Lu":0.861}
IDX = {m: i for i, m in enumerate(LAN)}
HET_O = {"O"}; HET_N = {"N"}; HET_CL = {"Cl"}


def read_xyz(p):
    L = open(p).read().splitlines(); n = int(L[0].split()[0])
    els, xyz = [], []
    for ln in L[2:2+n]:
        s = ln.split(); els.append(s[0]); xyz.append([float(v) for v in s[1:4]])
    return els, np.array(xyz, float)


def seed_counts(seeds_dir):
    sc = {}
    for sf in glob.glob(os.path.join(seeds_dir, "*.xyz")):
        try: sc[os.path.basename(sf)[:-4]] = int(open(sf).readline().split()[0])
        except Exception: pass
    return sc


def collect(dirpath, sc, donor, cut=2.9):
    """per-complex records: (metal, [generated donor distances])"""
    recs = []
    for f in glob.glob(os.path.join(dirpath, "**", "*.xyz"), recursive=True):
        seed = next((p for p in f.split(os.sep) if p in sc), None)
        if seed is None: continue
        ctx = sc[seed]
        try: els, xyz = read_xyz(f)
        except Exception: continue
        if len(els) <= ctx or els[0] not in LANSET: continue
        m = els[0]; d = np.linalg.norm(xyz - xyz[0], axis=1)
        ds = [d[i] for i in range(ctx, len(els)) if d[i] <= cut and els[i] in donor]
        if ds: recs.append((m, ds))
    return recs


def per_metal_r(recs, use_radius=False):
    per = {}
    for m, ds in recs:
        per.setdefault(m, []).extend(ds)
    xs, ys = [], []
    for m in LAN:
        if len(per.get(m, [])) >= 3:
            xs.append(RADIUS[m] if use_radius else IDX[m]); ys.append(np.mean(per[m]))
    if len(xs) < 3: return None, None, None, {}
    r = np.corrcoef(xs, ys)[0, 1]
    slope = np.polyfit(xs, ys, 1)[0]
    means = {m: np.mean(per[m]) for m in LAN if len(per.get(m, [])) >= 3}
    return r, slope, len(xs), means


def analyze(name, recs, B=3000):
    print(f"\n#### {name}  ({len(recs)} complexes with a generated donor) ####")
    r_idx, slope_idx, nm, means = per_metal_r(recs, use_radius=False)
    r_rad, slope_rad, _, _ = per_metal_r(recs, use_radius=True)
    if r_idx is None:
        print("  too few metals"); return
    order = [m for m in LAN if m in means]
    drop = means[order[0]] - means[order[-1]]
    print("  per-metal mean Ln-donor (A):", "  ".join(f"{m}:{means[m]:.3f}" for m in order))
    print(f"  metals={nm}  |  r(vs index)={r_idx:+.3f}  r(vs Shannon radius)={r_rad:+.3f}")
    print(f"  slope(vs index)={slope_idx*1000:+.1f} mAng/metal  |  {order[0]}->{order[-1]} drop = {drop*1000:+.0f} mAng")
    # bootstrap CI on r (resample complexes)
    br = []
    for _ in range(B):
        samp = [recs[random.randrange(len(recs))] for _ in recs]
        rr, _, k, _ = per_metal_r(samp)
        if rr is not None: br.append(rr)
    lo, hi = np.percentile(br, [2.5, 97.5])
    print(f"  bootstrap r 95%% CI = [{lo:+.3f}, {hi:+.3f}]  (B={B})")
    # PERMUTATION control: shuffle metal labels -> null r
    metals = [m for m, _ in recs]; dists = [ds for _, ds in recs]
    null = []
    for _ in range(B):
        random.shuffle(metals)
        rr, _, k, _ = per_metal_r(list(zip(metals, dists)))
        if rr is not None: null.append(rr)
    null = np.array(null)
    p = (null <= r_idx).mean()           # one-sided: how often chance gives r this negative
    print(f"  PERMUTATION null r: mean={null.mean():+.3f} sd={null.std():.3f}  ->  p(r<=obs) = {p:.4f}")
    print(f"     => observed r={r_idx:+.3f} is {abs(r_idx-null.mean())/max(null.std(),1e-9):.1f} sd below the metal-shuffled null")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_dir", required=True)
    ap.add_argument("--gen_dir", required=True)
    a = ap.parse_args()
    sc = seed_counts(a.seeds_dir)
    print("seeds:", len(sc))
    analyze("Ln-O  (generated donors)", collect(a.gen_dir, sc, HET_O))
    analyze("Ln-N  (generated donors)", collect(a.gen_dir, sc, HET_N))
    analyze("Ln-Cl (generated donors)", collect(a.gen_dir, sc, HET_CL))
