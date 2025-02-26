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
    python scripts/eval_train.py --uncond_diffusion_loadpath diffusion/defaults_H8_T100/uncond --diffusion_loadpath car_nearest/20250223-005933 --n_concepts 1 --seed ${seed}
done
