#!/bin/bash
#SBATCH --job-name=weight
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output="slurm/InvestESG/slurm-%J-%x.out"
python scripts/learn_concepts.py --n_concepts 4 --dataset_name "data/highway/training_dataset.pkl"  --uncond_diffusion_loadpath diffusion/defaults_H8_T100/uncond --diffusion_loadpath car_nearest/20250223-005933 --learn_conditions False --n_epochs 10