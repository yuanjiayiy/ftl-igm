#!/bin/bash
#SBATCH --job-name=highway
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output="slurm/InvestESG/slurm-%J-%x.out"
for seed in 1 2 3 4 5 6 7 8
do
    python scripts/eval_train.py --diffusion_loadpath diffusion/defaults_H8_T100/20250215-133903 --seed ${seed}
done
    