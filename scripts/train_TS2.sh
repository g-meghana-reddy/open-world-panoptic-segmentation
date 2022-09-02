#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --job-name=4DPLS_TS2
#SBATCH --time=3-12:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/4DPLS_TS2_%j.txt
#SBATCH --error=slurm_logs/4DPLS_TS2_%j.err

# module purge
# module load anaconda3-2019 
# module load gcc-6.3.0
# module load cuda-10.2 cudnn-10.2-v7.6.5.32

source activate 4dpls

# pre-training
python3 train_SemanticKitti.py -t 2 -s results/4DPLS_TS2_pretrain --pretrain --wandb

# training
# python3 train_SemanticKitti.py -t 2 -s results/4DPLS_TS2 -p 4DPLS_TS2_pretrain -i 3 --wandb