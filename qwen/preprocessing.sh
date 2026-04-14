#!/bin/bash
#SBATCH --job-name=dl-project-preprocess
#SBATCH -t 4:00:00              		        # Runtime in D-HH:MM
#SBATCH --mem-per-gpu 128G
#SBATCH --nodes=1 --ntasks-per-node=4 --gres=gpu:a100:2
#SBATCH --partition=ice-gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adibiasio3@gatech.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source .venv/bin/activate
export $(grep -v '^#' ../.env | xargs)

python preprocessing.py