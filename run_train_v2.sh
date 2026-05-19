#!/bin/bash
# Retraining with fixed GVP (no broken transfer_layer, lower dropout)
# Key changes vs original training:
#   - pretrained_weights removed → no randomly-initialized frozen transfer_layer
#   - drop_rate 0.5 → 0.1
#   - test_epochs 1 → 5 (reduce validation overhead)
#   - batch_size 64 (same as before)
#   - n_epochs 300 (more room for convergence)

cd /home/scli/diffusion_model/gvp_layer_Ln_LigandDiff

source /home/scli/anaconda3/etc/profile.d/conda.sh
conda activate ldtest

# Use whichever GPU has more free memory
export CUDA_VISIBLE_DEVICES=1

python -u train.py \
    --model gvp_dynamics \
    --data data/Ln_data \
    --train_data train \
    --val_data val \
    --hidden_nf 192 \
    --n_layers 5 \
    --drop_rate 0.1 \
    --attention False \
    --normalization_factor 100 \
    --aggregation_method sum \
    --normalization batch_norm \
    --diffusion_steps 500 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_loss_type l2 \
    --lr 1e-4 \
    --batch_size 64 \
    --n_epochs 300 \
    --test_epochs 5 \
    --tanh True \
    --exp_name GVP_fixed_v2 \
    --wandb_entity geometric \
    2>&1 | tee logs/train_v2_$(date +%Y%m%d_%H%M%S).log

echo "Training done."
