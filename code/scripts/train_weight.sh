#!/bin/bash
#SBATCH --job-name=car1
#SBATCH --partition=gpu-l40
#SBATCH --account=socialrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output="slurm/InvestESG/slurm-%J-%x.out"
python scripts/learn_concepts.py --n_concepts 5 --dataset_name "data/highway/test_dataset.pkl"  --loader datasets.HighwaySequencePartialObservedDataset --uncond_diffusion_loadpath diffusion/defaults_H8_T100/20250215-133903