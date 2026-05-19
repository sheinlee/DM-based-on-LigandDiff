#!/bin/bash
# Quick generation test on local GPU 1 (~8 GB free)
# Uses generate_test.py with BondedOct=False and relaxed connectivity

cd /home/scli/diffusion_model/gvp_layer_Ln_LigandDiff

source /home/scli/anaconda3/etc/profile.d/conda.sh
conda activate ldtest  # liganddiff has torch_cluster incompatible with 4090; ldtest has +pt22cu121

COMPLEX=data/Ln_data/val/EDUTUU_sub.xyz  # Tb N3O3 70 atoms, multi-dentate, no monodentate Cl
MODEL=models/config_scli_LigandDiff_bs64_date06-05_time13-59-32.201761/config_scli_LigandDiff_bs64_date06-05_time13-59-32.201761_epoch=84.ckpt
OUTDIR=test_4090_out

export CUDA_VISIBLE_DEVICES=1

python -u generate_test.py \
    --complex  $COMPLEX \
    --model    $MODEL \
    --outdir   $OUTDIR \
    --batch_size 8 \
    --n_samples 20 \
    --connectivity_thresh 0.8 \
    --atom_tol 2

echo "Done. Results in $OUTDIR"
