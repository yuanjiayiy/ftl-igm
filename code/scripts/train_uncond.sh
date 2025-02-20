#!/bin/bash
#SBATCH --job-name=uncond
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output="slurm/InvestESG/slurm-%J-%x.out"
python scripts/train.py  --loader datasets.HighwaySequencePartialObservedDataset --train_uncond True