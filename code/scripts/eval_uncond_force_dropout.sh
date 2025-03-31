#!/bin/bash
#SBATCH --job-name=uncond
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=19:00:00
#SBATCH --output="slurm/slurm-%J-%x.out"

# python scripts/eval_train.py \
#     --force_dropout True \
#     --diffusion_loadpath diffusion/defaults_H8_T100/20250317-173100 \
#     --eval_name 'unconditional w/o past traj'


python scripts/eval_train.py \
    --force_dropout True \
    --diffusion_loadpath diffusion/defaults_H8_T100/20250317-173100 \
    --eval_name 'unconditional w/o past traj check_crashed false' \
    --check_crashed False'
