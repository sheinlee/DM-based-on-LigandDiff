# DM-based-on-LigandDiff: Lanthanide Complex Ligand Generation

A 3D equivariant diffusion model for *de novo* ligand design in **lanthanide (Ln) coordination complexes**, adapted from [LigandDiff](https://github.com/Neon8988/LigandDiff) (transition metals) and extended to the full La–Lu series with support for variable coordination numbers (CN=4–12).

---

## Headline result: generated structures reproduce the lanthanide contraction

The strongest evidence that the model has learned real f-block coordination chemistry — not just
RDKit-valid molecular graphs — is that the **geometry** of its generated complexes is physically
correct across the entire La→Lu series. Three independent, geometry-level checks (all on `ep276`):

**1. Bond length — the lanthanide contraction.**
The mean **Ln–O donor distance** of *generated* ligand donors decreases monotonically across the
series, **La ≈ 2.48 Å → Lu ≈ 2.14 Å**, reproducing the lanthanide contraction.

- Pearson **r = −0.85** — measured on **generated donors only** (context/scaffold donors excluded,
  so the trend is the model's own atom placement, not retained scaffold).
- bootstrap 95 % CI **[−0.92, −0.63]**; slope **−19 mÅ / metal**; La→Lu drop **0.34 Å**; n = 286.
- **Permutation control** (shuffle the metal labels): metal-shuffled null r = 0.0 ± 0.35, giving
  **p = 3×10⁻⁴**. The metal-dependence is a *learned* behaviour, not a measurement artefact.
- Experimental CSD reference: r = −0.974.
- Ln–N shows the same direction but is not significant (r = −0.61, p = 0.054); Ln–Cl too sparse (n=9).

**2. Coordination-angle distribution.**
The full **donor–M–donor angle distribution** of generated complexes matches the CSD training
reference to **Wasserstein-1 = 1.2°** (generated 98.1° ± 32.7 vs experimental 99.1° ± 32.0;
29,731 vs 326,033 angles). The characteristic bimodal shape — adjacent donors ~70–85°, near-*trans*
~130–145° — is reproduced bin-for-bin. The model gets the coordination **polyhedron**, not just the
radial distances. Angles are scale-free and unaffected by scaffold conditioning, so this is the
cleanest geometry test.

**3. Coordination number.**
The generated first-shell CN distribution (median **8**, mean 7.73) tracks the experimental high-CN
preference of lanthanides (median 8, mean 7.44; CN 8–10 dominate). Because the generated complexes
are seed-conditioned on CN 8–10 scaffolds, CN is a *consistency* check rather than a free prediction
— the angle result above is the unconditioned geometry test.

> **Report RAW geometry, not force-field-relaxed.** These trends live in the **raw model output**.
> Standard xtb **GFN-FF** relaxation *destroys* the contraction (r collapses −0.94 → −0.13, Ln–O
> shortened by ~0.17 Å) because GFN-FF's f-in-core parameters are unreliable for the lanthanides. Any
> downstream relaxation of f-element structures should use an f-aware method (RECP-DFT / an
> Ln-specific xTB), **not** GFN-FF. This is a practical caution for anyone generating f-block structures.

Reproduce (both scripts are `argparse`-driven; point them at your generated xyz tree):

```bash
# Bond-length contraction: r, bootstrap CI, permutation control, Shannon-radius axis
python eval/ln_contraction_stats.py --seeds_dir <held-out seeds> --gen_dir <raw generated xyz>
# Coordination-angle W1 distance + first-shell CN distribution vs the training reference
python eval/ln_geom_extra.py        --train_pt  <Ln train.pt>    --gen_dir <raw generated xyz>
```

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

## Held-out evaluation (V / U / N + yield + recall)

Beyond the qualitative high-CN checks above, the model was scored on a **held-out generative
benchmark**: 40 held-out Ln seeds (CN 8–10), save-all generation, xtb GFN-FF relaxation, then the
standard per-ligand **Validity / Uniqueness / Novelty** trio plus a complex-level **Yield** and a
whole-complex **topology Recall**. The held-out reference (`Ln_data_new/val`), the novelty reference,
and the metric code are the *same* ones used for the coordination-site-conditioned `Multi` model, so
every row is directly comparable. Full numbers:
[`results/heldout_eval_CN8-10.txt`](results/heldout_eval_CN8-10.txt).

**Our strict metric** (RDKit `rdDetermineBonds` on xtb-relaxed geometry). `ep276` = the
connectivity-fine-tuned checkpoint (`dconn3_from274_conn_epoch=276`) used across this README.

| model | training data | Validity | Novelty (graph) | Yield | Recall | partial-overlap |
|-------|---|:---:|:---:|:---:|:---:|:---:|
| **ep276** (conn-finetune, big data) | Ln_data_new (14,478, CN 4–12) | 62.9 % | 86.2 % | **10.0 %** | 0/24 | 0.611 |
| **transfer** (GVP, TM→Ln, ep134) | TM pre-train → Ln fine-tune | 54.0 % | 88.2 % | 4.9 % | 0/24 | 0.473 |
| baseline (GVP 192/5) | Ln (12,738) | 60.4 % | 84.1 % | 5.5 % | 0/24 | 0.611 |
| B (GVP 256/7, capacity) | Ln (12,738) | 57.9 % | 87.1 % | 3.7 % | 0/24 | 0.528 |
| Multi (192/6, **coord-site**) | Ln (12,738) | **68.0 %** | **95.6 %** | 9.9 % | 0/27 | 0.646 |

**Literature metric** (OpenBabel + molSimplify ligand split — reproduces how LigandDiff scores its
own numbers; d-block reference rows for context):

| model | Validity | Connectivity | Uniqueness | Novelty |
|-------|:---:|:---:|:---:|:---:|
| **ep276** | 100 %¹ | 76.1 % | 60.7 % | 96.3 % |
| **transfer** | 100 %¹ | 74.2 % | 70.9 % | 96.8 % |
| *LigandDiff (d-block)* | *94 %* | *96 %* | *97 %* | *96 %* |
| *multi-LigandDiff (d-block)* | *99 %* | *99 %* | *96 %* | *100 %* |

¹ Validity is *degenerate* at 100 % for the single-ligand Ln model (OpenBabel `build_mol` almost
always yields a sanitizable molecule) → not informative here; **connectivity** is the diagnostic.

- **The connectivity fine-tune matters.** `ep276` (Validity 62.9 %, **Yield 10.0 %**) clearly beats
  its `ep274` base (59.8 % / 4.9 %) — yield roughly *doubled*, reaching ≈ the coordination-site-
  conditioned `Multi` (9.9 %). Bigger data + connectivity fine-tuning closes the **yield** gap.
- **`Multi` still leads overall** (Validity 68.0 %, Novelty 95.6 %) — telling the model *where the
  coordination sites are* helps beyond data / capacity / transfer.
- **Staged TM→Ln transfer underperforms** (Validity 54.0 %, lowest fidelity 0.473; undertrained).
- Under the literature algorithm the models reach **novelty ~96 %** (in the literature band); the real
  gap is **connectivity ~75 %** vs LigandDiff's 96 % (harder f-block domain + a metric that is fragile
  for coordination ligands, see the note above). All models: **0 % exact topology recall**.

> See the companion
> [Multi-Liganddiff-on-Ln-complexes](https://github.com/sheinlee/Multi-Liganddiff-on-Ln-complexes)
> repo for the full metric-algorithm analysis and the combined-TM+Ln study.

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
