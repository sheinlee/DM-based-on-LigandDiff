"""
cif_to_dataset.py  –  CIF → xyz + train_smiles.csv

Key design: reads ONLY the asymmetric unit from _atom_site_* fields
(fractional coords converted to Cartesian), avoiding openbabel's
symmetry expansion that duplicates atoms.

Filters:
  - Exactly 1 Ln metal (La-Lu) in asymmetric unit
  - No other transition/alkali metals
  - 7 <= heavy atoms <= 200
  - molSimplify ligand_breakdown CN in 4-12

Usage:
  python cif_to_dataset.py --cif_dir /path/to/cif_output --outdir xyz_ln_all
"""

import argparse, math, os, re, sys, tempfile, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/scli/diffusion_model/gvp_layer_Ln_LigandDiff')

from openbabel import openbabel as ob
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown

LN_Z      = set(range(57, 72))
OTHER_M_Z = (set(range(21, 31)) | set(range(39, 49)) |
             set(range(72, 81)) | {19, 20, 37, 38, 55, 56})
VALID_CN  = set(range(4, 13))

# Element symbol → atomic number
_SYM2Z = {}
def sym2z(sym):
    global _SYM2Z
    if not _SYM2Z:
        import re
        _conv = ob.OBConversion()
        for z in range(1, 104):
            s = ob.GetSymbol(z)
            if s:
                _SYM2Z[s.capitalize()] = z
    return _SYM2Z.get(sym.capitalize(), 0)


# ── CIF asymmetric-unit parser ─────────────────────────────────────────────

def parse_cell(text):
    """Extract unit cell a,b,c,α,β,γ from CIF text."""
    def grab(tag):
        m = re.search(rf'^{tag}\s+([\d.]+)', text, re.M)
        return float(m.group(1)) if m else None
    a = grab('_cell_length_a') or grab('_cell\.length_a')
    b = grab('_cell_length_b') or grab('_cell\.length_b')
    c = grab('_cell_length_c') or grab('_cell\.length_c')
    al = grab('_cell_angle_alpha') or grab('_cell\.angle_alpha') or 90.0
    be = grab('_cell_angle_beta')  or grab('_cell\.angle_beta')  or 90.0
    ga = grab('_cell_angle_gamma') or grab('_cell\.angle_gamma') or 90.0
    return a, b, c, al, be, ga


def frac_to_cart(fx, fy, fz, a, b, c, al, be, ga):
    """Convert fractional → Cartesian coordinates."""
    al, be, ga = math.radians(al), math.radians(be), math.radians(ga)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sg = math.sin(ga)
    v = math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2*ca*cb*cg)
    x = a*fx + b*cg*fy + c*cb*fz
    y = b*sg*fy + c*(ca - cb*cg)/sg*fz
    z = c*v/sg*fz
    return x, y, z


def _clean_val(s):
    """Remove CIF uncertainty suffix: '2.345(3)' → '2.345'."""
    return re.sub(r'\(.*?\)', '', s)


def parse_atoms(text):
    """
    Parse _atom_site_* loop from CIF text.
    Returns list of (element_symbol, x_cart, y_cart, z_cart).
    """
    a, b, c, al, be, ga = parse_cell(text)
    if not all([a, b, c]):
        return []

    # Find the atom_site loop block
    block_match = re.search(r'loop_\s*((?:_atom_site_\S+\s+)+)((?:(?!loop_|data_)[\s\S])*)',
                            text, re.M)
    if not block_match:
        return []

    header_str = block_match.group(1)
    data_str   = block_match.group(2)

    headers = re.findall(r'_atom_site_\S+', header_str)
    # Map column names to indices
    col = {h.lower(): i for i, h in enumerate(headers)}

    # Determine element column (type_symbol preferred over label)
    elem_col = col.get('_atom_site_type_symbol',
               col.get('_atom_site_type_symbol'.lower(),
               col.get('_atom_site_label', None)))
    x_col = col.get('_atom_site_fract_x',
            col.get('_atom_site_cartn_x', None))
    y_col = col.get('_atom_site_fract_y',
            col.get('_atom_site_cartn_y', None))
    z_col = col.get('_atom_site_fract_z',
            col.get('_atom_site_cartn_z', None))

    use_frac = '_atom_site_fract_x' in col
    if None in (elem_col, x_col, y_col, z_col):
        return []

    # Parse rows (handle multi-line / whitespace-delimited)
    tokens = data_str.split()
    n_cols = len(headers)
    atoms  = []
    i = 0
    while i + n_cols <= len(tokens):
        row = tokens[i:i + n_cols]
        i  += n_cols
        # Skip disorder / partial occupancy markers and comment lines
        if row[0].startswith('#') or row[0].startswith('_'):
            i -= n_cols - 1
            continue

        raw_elem = row[elem_col]
        # Extract element symbol (may be like 'Ce1', 'O2a')
        m = re.match(r'([A-Z][a-z]?)', raw_elem)
        if not m:
            continue
        elem = m.group(1)

        try:
            fx = float(_clean_val(row[x_col]))
            fy = float(_clean_val(row[y_col]))
            fz = float(_clean_val(row[z_col]))
        except (ValueError, IndexError):
            continue

        if use_frac:
            x, y, z = frac_to_cart(fx, fy, fz, a, b, c, al, be, ga)
        else:
            x, y, z = fx, fy, fz
        atoms.append((elem, x, y, z))

    return atoms


# ── Fragment extraction ────────────────────────────────────────────────────

def atoms_to_fragment_xyz(atoms):
    """
    From asymmetric-unit atom list, build xyz string of just the
    Ln-containing connected component (no H). Returns None if invalid.
    """
    # Keep only heavy atoms with known Z
    heavy = [(e, x, y, z) for e, x, y, z in atoms
             if e != 'H' and e != 'D' and sym2z(e) > 0]
    if not heavy:
        return None

    # Find Ln atoms
    ln_idx = [i for i, (e,_,_,_) in enumerate(heavy) if sym2z(e) in LN_Z]
    if len(ln_idx) != 1:
        return None

    # Build distance-based adjacency (covalent radii × 1.35 cutoff)
    COV = {  # Å, approximate
        'H':0.31,'C':0.77,'N':0.71,'O':0.66,'S':1.02,'P':1.07,'F':0.57,
        'Cl':1.02,'Br':1.14,'I':1.33,'B':0.82,'Si':1.17,'Se':1.20,
        'La':1.69,'Ce':1.63,'Pr':1.76,'Nd':1.74,'Pm':1.73,'Sm':1.72,
        'Eu':1.68,'Gd':1.69,'Tb':1.68,'Dy':1.67,'Ho':1.66,'Er':1.65,
        'Tm':1.64,'Yb':1.70,'Lu':1.62,
    }
    DEFAULT = 1.50

    def threshold(e1, e2):
        return (COV.get(e1, DEFAULT) + COV.get(e2, DEFAULT)) * 1.35

    n = len(heavy)
    adj = [[] for _ in range(n)]
    for i in range(n):
        e1, x1, y1, z1 = heavy[i]
        for j in range(i+1, n):
            e2, x2, y2, z2 = heavy[j]
            d = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            if d < threshold(e1, e2):
                adj[i].append(j)
                adj[j].append(i)

    # BFS from Ln
    start = ln_idx[0]
    visited, queue = set(), [start]
    while queue:
        v = queue.pop()
        if v in visited:
            continue
        visited.add(v)
        queue.extend(adj[v])

    frag = [heavy[i] for i in sorted(visited)]

    # Reject if other metals present
    for e, *_ in frag:
        if sym2z(e) in OTHER_M_Z:
            return None

    if not (7 <= len(frag) <= 200):
        return None

    # Build xyz: Ln first
    ln_entry = next((e,x,y,z) for e,x,y,z in frag if sym2z(e) in LN_Z)
    others   = [(e,x,y,z) for e,x,y,z in frag if sym2z(e) not in LN_Z]
    lines    = [str(len(frag)), '']
    for e, x, y, z in [ln_entry] + others:
        lines.append(f'{e:2s} {x:14.8f} {y:14.8f} {z:14.8f}')
    return '\n'.join(lines)


# ── CN check ──────────────────────────────────────────────────────────────

def get_cn(xyz_str, label):
    """Return true coordination number = number of atoms bonded to metal."""
    with tempfile.NamedTemporaryFile(suffix='.xyz', mode='w', delete=False) as tmp:
        lns = xyz_str.split('\n')
        tmp.write(lns[0]+'\n'+label+'\n'+'\n'.join(lns[2:])+'\n')
        path = tmp.name
    try:
        mol = mol3D()
        mol.readfromxyz(path)
        metal_idx = mol.findMetal()[0]
        bonded = mol.getBondedAtomsSmart(metal_idx, oct=False)
        return len(bonded)
    except Exception:
        return -1
    finally:
        os.unlink(path)


# ── SMILES extraction ─────────────────────────────────────────────────────

def extract_ligand_smiles(xyz_str, label):
    lns = xyz_str.split('\n')
    ligand_lines = lns[2+1:]          # skip n_atoms, label, metal line
    if not ligand_lines:
        return []
    n = len(ligand_lines)
    mini_xyz = f'{n}\n\n' + '\n'.join(ligand_lines) + '\n'

    conv_in  = ob.OBConversion(); conv_in.SetInFormat('xyz')
    conv_smi = ob.OBConversion(); conv_smi.SetOutFormat('smi')
    mol = ob.OBMol(); mol.SetAutomaticPartialCharge(False)
    if not conv_in.ReadString(mol, mini_xyz):
        return []
    mol.ConnectTheDots(); mol.PerceiveBondOrders()

    smiles_list = []
    for frag in mol.Separate():
        smi_line = conv_smi.WriteString(frag).strip()
        smi = smi_line.split()[0] if smi_line else ''
        if smi:
            smiles_list.append(smi)
    return smiles_list


# ── Per-directory processing ───────────────────────────────────────────────

def process_dir(cif_dir, outdir, smiles_set, counters, max_structs):
    for fname in sorted(os.listdir(cif_dir)):
        if not fname.endswith('.cif') or counters['saved'] >= max_structs:
            break
        label = fname.replace('.cif', '')
        counters['total'] += 1

        with open(os.path.join(cif_dir, fname), 'r', errors='replace') as f:
            text = f.read()

        atoms = parse_atoms(text)
        if not atoms:
            counters['skip_parse'] = counters.get('skip_parse', 0) + 1
            continue

        xyz = atoms_to_fragment_xyz(atoms)
        if xyz is None:
            counters['skip_struct'] += 1
            continue

        cn = get_cn(xyz, label)
        if cn not in VALID_CN:
            counters['skip_cn'] += 1
            if 'cn_bad' not in counters:
                counters['cn_bad'] = {}
            counters['cn_bad'][cn] = counters['cn_bad'].get(cn, 0) + 1
            continue

        lns = xyz.split('\n')
        with open(os.path.join(outdir, f'{label}_sub.xyz'), 'w') as f:
            f.write(lns[0]+'\n'+label+'\n'+'\n'.join(lns[2:])+'\n')
        counters['saved'] += 1
        if 'cn_dist' not in counters:
            counters['cn_dist'] = {}
        counters['cn_dist'][cn] = counters['cn_dist'].get(cn, 0) + 1

        for smi in extract_ligand_smiles(xyz, label):
            smiles_set.add(smi)

        if counters['saved'] % 500 == 0 or counters['saved'] <= 3:
            print(f"  [{counters['total']:6d}] saved={counters['saved']:5d}  "
                  f"skip_struct={counters['skip_struct']}  "
                  f"skip_cn={counters['skip_cn']}  "
                  f"cn_dist={counters.get('cn_dist',{})}",
                  flush=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cif_dir',    default='/home/scli/diffusion_model/cif_output')
    p.add_argument('--outdir',     default='/home/scli/diffusion_model_new/xyz_ln_all')
    p.add_argument('--max_structs',type=int, default=30000)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    smiles_set = set()
    counters   = {'total': 0, 'saved': 0, 'skip_struct': 0, 'skip_cn': 0}

    elem_dirs = sorted([os.path.join(args.cif_dir, d)
                        for d in os.listdir(args.cif_dir)
                        if os.path.isdir(os.path.join(args.cif_dir, d))])
    print(f'Found {len(elem_dirs)} element dirs, processing...')

    for elem_dir in elem_dirs:
        if counters['saved'] >= args.max_structs:
            break
        elem = os.path.basename(elem_dir)
        n_cif = len([f for f in os.listdir(elem_dir) if f.endswith('.cif')])
        print(f'\n{elem}: {n_cif} CIF files', flush=True)
        process_dir(elem_dir, args.outdir, smiles_set, counters, args.max_structs)

    smiles_path = os.path.join(args.outdir, 'train_smiles.csv')
    with open(smiles_path, 'w') as f:
        f.write('\n'.join(sorted(smiles_set)) + '\n')

    print(f'\n=== Done ===')
    print(f'  CIF read       : {counters["total"]}')
    print(f'  Saved          : {counters["saved"]}')
    print(f'  Skip (struct)  : {counters["skip_struct"]}')
    print(f'  Skip (CN)      : {counters["skip_cn"]}')
    print(f'  Bad CN values  : {counters.get("cn_bad",{})}')
    print(f'  CN distribution: {counters.get("cn_dist",{})}')
    print(f'  Unique SMILES  : {len(smiles_set)}')
    print(f'  Output         : {args.outdir}/')


if __name__ == '__main__':
    main()
