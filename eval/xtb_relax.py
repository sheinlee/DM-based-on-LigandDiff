"""
Relax generated coordination-complex .xyz with xtb, so bond lengths are cleaned up BEFORE
topology perception (recall is topology-based; geometry is relaxable — user's rubric).

Lanthanide caveat: GFN2-xTB lacks reliable f-element parameters, so the DEFAULT method here is
**GFN-FF** (a force field with broad element coverage). Optionally --freeze_metal pins the metal
(atom 0) and relaxes only the ligand framework — which is exactly where the bond-length deficit
is (Ln–donor bonds were already good; ligand-internal bonds were off) and avoids metal-param
artifacts. Charge is a per-complex quantity (Ln3+ + ligand charges); a single --charge is an
approximation, but for GEOMETRY cleanup (not energetics) GFN-FF is robust to it.

Run in a conda env that has xtb installed:
  python xtb_relax.py --in_dir <gen_dir> --out_dir <relaxed_dir> [--charge 0] [--freeze_metal] [--method gfnff]
Then run eval_topology_recall.py --gen_dir <relaxed_dir>.
"""
import os, glob, argparse, shutil, subprocess, tempfile


def relax_one(xyz, out_xyz, charge, method, freeze_metal, timeout, omp=1):
    # CRITICAL: pin OMP threads. Unpinned, xtb grabs all cores and thread-thrashes on a shared
    # node (a 28-atom GFN-FF opt went from 0.11s -> >120s). omp=1 lets you run many in parallel.
    env = dict(os.environ, OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp), OMP_STACKSIZE='1G')
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(xyz, os.path.join(td, 'in.xyz'))
        cmd = ['xtb', 'in.xyz', '--opt', 'crude', '--' + method, '--chrg', str(charge)]
        if freeze_metal:
            # xtb uses 1-based atom indexing; atom 0 (the metal) -> "1"
            with open(os.path.join(td, 'xcontrol'), 'w') as f:
                f.write('$fix\n  atoms: 1\n$end\n')
            cmd += ['--input', 'xcontrol']
        try:
            # NOTE: DEVNULL (not capture_output=) — xtbenv's python is old (<3.7); capture_output raises.
            subprocess.run(cmd, cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return False
        opt = os.path.join(td, 'xtbopt.xyz')
        if os.path.exists(opt):
            shutil.copy(opt, out_xyz)
            return True
        return False


def main(a):
    files = sorted(glob.glob(os.path.join(a.in_dir, '**', '*.xyz'), recursive=True))
    ok = fail = 0
    for f in files:
        out = os.path.join(a.out_dir, os.path.relpath(f, a.in_dir))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            if relax_one(f, out, a.charge, a.method, a.freeze_metal, a.timeout, a.omp):
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    print(f'xtb-relaxed {ok} / {ok + fail}  ({fail} failed)  method={a.method} freeze_metal={a.freeze_metal}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--charge', type=int, default=0)
    p.add_argument('--method', default='gfnff', help='gfnff (broad coverage, default) | gfn2 (no f-block)')
    p.add_argument('--freeze_metal', action='store_true', help='pin atom 0 (metal); relax only ligands')
    p.add_argument('--timeout', type=int, default=120, help='seconds per complex')
    p.add_argument('--omp', type=int, default=1, help='OMP threads per xtb (keep small so many can run in parallel)')
    main(p.parse_args())
