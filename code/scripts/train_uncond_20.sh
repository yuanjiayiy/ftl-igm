#!/bin/bash
#SBATCH --job-name=uncond
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=19:00:00
#SBATCH --output="slurm/slurm-%J-%x.out"
python scripts/train.py --dataset_path data/highway_1_2/demos.pkl --force_dropout True