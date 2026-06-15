#!/bin/bash
#SBATCH -J simvla_libero
#SBATCH -A cis260130p    
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:4           # bf16-capable; or h100-80:4, or v100-32:4 (+fp16)
#SBATCH -t 48:00:00                # 48h is the GPU partition max
#SBATCH -o logs/simvla_%j.out
#SBATCH -e logs/simvla_%j.err

set -e
export PYTHONNOUSERSITE=1
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
conda activate simvla
module load cuda
cd ~/SimVLA
# bash train_smolvlm_small.sh
bash train_smolvlm_large.sh
