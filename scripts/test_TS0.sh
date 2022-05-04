#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --job-name=test_4DPLS_TS0
#SBATCH --time=1-00:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=slurm_logs/test_4DPLS_TS0_%j.txt
#SBATCH --error=slurm_logs/test_4DPLS_TS0_%j.err

source activate 4dpls
python3 test_models.py -t 0 -c results/4DPLS_TS0
# results/4DPLS_TS0