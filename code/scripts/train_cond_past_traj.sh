#!/bin/bash
#SBATCH --job-name=cond_329
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=19:00:00
#SBATCH --output="slurm/slurm-%J-%x.out"
python scripts/train_conditional.py \
    --dataset_path data/highway_1_2/demos.pkl \
    --force_dropout False \
    --frozen_unconditional_model_path /mmfs1/gscratch/cse/jiayiy9/ftl-igm/code/logs/highway/diffusion/defaults_H8_T100/20250329-140003