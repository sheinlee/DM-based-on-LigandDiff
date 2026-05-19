#!/bin/bash
#SBATCH -J bond_gnn
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -o logs/bond_gnn_%j.out
#SBATCH -e logs/bond_gnn_%j.err

cd /home/scli/diffusion_model/gvp_layer_Ln_LigandDiff
echo "=== Bond GNN training started $(date) ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

/home/scli/anaconda3/envs/ldtest/bin/python3 -u src/bond_gnn/train.py \
    --xyz_dir /home/scli/diffusion_model_new/xyz_ln_all \
    --cif_dir /home/scli/diffusion_model/cif_output \
    --out      src/bond_gnn/bond_gnn.pt \
    --epochs   80 \
    --batch_size 32 \
    --hidden_dim 128 \
    --n_layers 4 \
    --lr 1e-3

echo "=== Done $(date) ==="
