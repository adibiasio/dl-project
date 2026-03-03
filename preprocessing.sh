#!/bin/bash
#SBATCH --job-name=dl-project-preprocess
#SBATCH -t 4:00:00              		# Runtime in D-HH:MM
#SBATCH --mem-per-gpu 128G
#SBATCH -n 1                          # number of CPU cores
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -C "H100"

export HF_TOKEN=XXX
python blip2_preprocessing.py