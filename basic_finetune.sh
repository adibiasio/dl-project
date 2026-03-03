#!/bin/bash
#SBATCH --job-name=dl-project
#SBATCH -t 8:00:00              		# Runtime in D-HH:MM
#SBATCH -n 8                          # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "H100"

export HF_TOKEN=XX
source .venv/bin/activate
python basic_finetune_blip2.py