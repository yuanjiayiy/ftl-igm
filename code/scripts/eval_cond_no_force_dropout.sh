#!/bin/bash
#SBATCH --job-name=cond
#SBATCH --partition=gpu-l40s
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=19:00:00
#SBATCH --output="slurm/slurm-%J-%x.out"
conditional_model_path='diffusion/defaults_H8_T100/20250330-134404'
frozen_unconditional_model_path='/mmfs1/gscratch/cse/jiayiy9/ftl-igm/code/logs/highway/diffusion/defaults_H8_T100/20250329-140003'
# python scripts/eval_train.py --force_dropout True \
#     --diffusion_loadpath ${conditional_model_path} \
#     --frozen_unconditional_model_path ${frozen_unconditional_model_path} \
#     --eval_conditional_model True \
#     --eval_name 'conditional w/ past traj'

python scripts/eval_train.py --force_dropout True \
    --diffusion_loadpath ${conditional_model_path} \
    --frozen_unconditional_model_path ${frozen_unconditional_model_path} \
    --eval_conditional_model True \
    --eval_name 'conditional w/ past traj check_crashed false' \
    --check_crashed False