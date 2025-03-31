#!/bin/bash
#SBATCH --job-name=random
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=19:00:00
#SBATCH --output="slurm/slurm-%J-%x.out"

# python scripts/eval_train.py \
#     --diffusion_loadpath /mmfs1/gscratch/cse/jiayiy9/ftl-igm/code/logs/highway/diffusion/defaults_H8_T100/20250325-111216 \
#     --eval_name 'random weight'

python scripts/eval_train.py \
    --diffusion_loadpath /mmfs1/gscratch/cse/jiayiy9/ftl-igm/code/logs/highway/diffusion/defaults_H8_T100/20250325-111216 \
    --eval_name 'random weight check_crashed false' \
    --check_crashed False
