# DM-based-on-LigandDiff: Lanthanide Complex Ligand Generation

A 3D equivariant diffusion model for *de novo* ligand design in **lanthanide (Ln) coordination complexes**, adapted from [LigandDiff](https://github.com/Neon8988/LigandDiff) (transition metals) and extended to the full La–Lu series with support for variable coordination numbers (CN=4–12).

---

## Latest Progress (2026-06)

Validation of high-CN generation on the full CN 4–12 training distribution (`Ln_data_new`; verified true-CN spans 0–12, peaks at CN 8/9; La: n=397, mean 7.4), using the connectivity-aware fine-tuned checkpoint **ep276**:

- **Real CN9 La scaffold → 51/52** generated complexes are clean, chemically-meaningful 9-coordinate La (9 O donors @ 2.5–2.6 Å).
- **CN12 La scaffold → 53** clean CN 11–14 La complexes.
- **~17%** of connected generated ligands are topologically identical (Tanimoto = 1.0) to known CSD ligands — e.g. piperidine (`WEBMOF`), (cyclohexylmethyl)amine (`KAWYAK`), and the characteristic Ln oxo-anions nitrate / carbonate / sulfate.

**Evaluation beyond RDKit validity.** A generated complex is accepted only if: the *true* coordination number (O/N donors within real bonding distance) is metal-appropriate, all ligands are connected, there is no atomic clash, and it passes a 2-stage **xtb** (GFN-FF → GFN2, Ln→La closed-shell cap) DFT-readiness test. Clean organic-chelate complexes (e.g. CN8 Yb) relax intact (mean Δlargest_frac +0.085, 6/6 GFN2-converged); aqua/oxo complexes need explicit H before QM.

**Metric notes.**
- *Connectivity* on metal-stripped, force-merged ligands is mis-specified for coordination complexes (ground truth itself scores conn ≈ 4%) — use as a relative diagnostic only.
- *molSimplify CN* (`BondedOct=False`) over-counts phantom ring-carbons at ~2.7 Å (≈ +3): a real CN9 = 6 true N donors @ 2.31–2.38 Å + 3 phantom C @ 2.71–2.75 Å. True CN is measured by donor bonding distance, and the phantom carbons relax away under xtb.

**Checkpoint selection.** ep276 was selected by a connectivity-aware criterion (validation VLB anti-correlates with connectivity).

---

## Key Changes from Original LigandDiff

| Component | Original LigandDiff | This Work |
|---|---|---|
| Target metals | Transition metals (Cr,Mn,Fe,Co,Ni,Cu,Zn,Ru,Pd) | **Full Ln series (La–Lu)** |
| Coordination number | 6 (octahedral only) | **CN 4–12** |
| Network | EGNN | **GVP (Geometric Vector Perceptron) + EGNN** |
| Training data | ~17k TM complexes | **14,478 mononuclear Ln complexes (CSD 2023)** |
| Bond detection | openbabel distance-only | **Bond GNN (MPNN, F1=0.948) + valence penalty** |

---

## Dataset

**Source**: Cambridge Structural Database (CSD 2023)  
**Filter**: Mononuclear Ln complexes, CN=4–12, heavy atoms 7–200 (H removed)  
**Size**: 14,478 structures → **42,521 ligand-level training samples**  
**Elements supported**: C, N, O, S, Br, Cl, P, F (8 atom types); metals La–Lu (all 14 Ln)

### Reproducing the dataset

```bash
# Step 1: CIF → xyz (reads _geom_bond_* for true bond labels, supports CN 4-12)
python data/cif_to_dataset.py \
    --cif_dir /path/to/csd_cif_output \
    --outdir  xyz_ln_all \
    --max_structs 30000

# Step 2: xyz → train.pt / val.pt
python data/xyz_to_pt.py \
    --xyz_dir xyz_ln_all \
    --out_dir data/Ln_data_new \
    --val_split 0.1
```

The `data/Ln_data/train/` and `val/` directories contain the original 887 six-coordinate Ln complexes used in preliminary experiments (BondedOct=True).

---

## Architecture

```
Input complex (.xyz)
    │
    ├─ Metal + fixed ligands  ──→  context (frozen)
    └─ Target ligand atoms    ──→  diffuse with GVP dynamics

GVP Dynamics (5 layers, hidden_nf=192):
  GVPLayerNorm → GVP(in=0 vectors) → GVPConvLayer ×5 → embedding_out
  ↓
  eps_hat (predicted noise for positions + atom types)
  ↓
  x_pred = (z_t - σ_t × eps_hat) / α_t
  ↓
  Soft valence penalty: Σ relu(soft_valence(O) - 2)²  [differentiable]

Post-processing:
  Bond GNN (3–4 layer MPNN) → chemically valid bond assignment
  Valence correction → remove longest bond if atom exceeds max valence
```

---

## Training

```bash
# Full training from scratch (42,521 samples, CN 4-12)
python train.py --config config_v2.yml

# Fine-tune with soft valence penalty (from epoch 216 checkpoint)
python train.py --config gvp_Ln_full/config_valence.yml
```

Key config parameters (`config_v2.yml`):
```yaml
model: gvp_dynamics
hidden_nf: 192
n_layers: 5
drop_rate: 0.1
diffusion_steps: 500
diffusion_noise_schedule: polynomial_2
batch_size: 16
valence_lambda: 0.5   # soft valence penalty weight
```

### Bond GNN (post-processing)

Trained on CIF crystallographic bond labels (precision=0.916, recall=0.983, F1=0.948).  
Replaces openbabel's distance-only bond detection for generated ligand atoms.

```bash
# Train Bond GNN
python src/bond_gnn/train.py \
    --xyz_dir xyz_ln_all \
    --cif_dir /path/to/csd_cif_output \
    --out     src/bond_gnn/bond_gnn.pt \
    --epochs  80
```

---

## Generation

```bash
# Generate new ligands for a Ln complex (supports CN 4-12)
python generate_test.py \
    --complex  data/AGIZEZ_sub.xyz \          # CN=9 Nd example
    --model    models/.../epoch=216.ckpt \
    --outdir   generated_out \
    --n_samples 20 \
    --connectivity_thresh 0.5 \
    --atom_tol 5
```

**Example results** (epoch=216, CN=9 Nd complex, 3×tridentate):
- valid_ligand: 100%
- connected (≥50%): 17%  
- Saved complexes: 9/60
- Atom position range: 1–25 Å (physically reasonable)

---

## molSimplify Extension

The `molSimplify/` directory extends the original [molSimplify](https://github.com/hjkgrp/molSimplify) with full lanthanide support:
- Covalent/VDW radii for La–Lu
- Bond distance thresholds for Ln–O/N bonds
- Metal recognition: `findMetal(transition_metals_only=True)` now finds Ln atoms
- `ligand_breakdown(BondedOct=False)` supports any coordination number

---

## Attribution

This work builds on:
- **LigandDiff** — [Neon8988/LigandDiff](https://github.com/Neon8988/LigandDiff), MIT License, Copyright 2024 Hongni Jin
- **DiffHopp GVP** — [jostorge/diffusion-hopping](https://github.com/jostorge/diffusion-hopping), MIT License, Copyright 2022 Jos Torge, Charles Harris, Simon Mathis
- **molSimplify** — [hjkgrp/molSimplify](https://github.com/hjkgrp/molSimplify)
- **CSD 2023** — Cambridge Structural Database, Cambridge Crystallographic Data Centre
