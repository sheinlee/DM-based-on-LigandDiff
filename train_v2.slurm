#!/bin/bash
#SBATCH -J LnDiff_GVP_v2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err

set -euo pipefail

cd /home/scli/diffusion_model/gvp_layer_Ln_LigandDiff

source /home/scli/anaconda3/etc/profile.d/conda.sh
conda activate ldtest

echo "=== Job ${SLURM_JOB_ID} started on $(hostname) at $(date) ==="
echo "GPU(s) allocated: ${CUDA_VISIBLE_DEVICES:-not set by SLURM}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

python -u train.py --config config_v2.yml

echo "=== Training finished at $(date) ==="
