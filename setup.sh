#!/bin/bash
#SBATCH --job-name=dl-project-setup
#SBATCH -t 00:30:00              		# Runtime in D-HH:MM
#SBATCH -n 12                          # number of CPU cores
#SBATCH -N 1

uv venv
source .venv/bin/activate
uv sync
export KAGGLE_API_TOKEN=XXX
kaggle datasets download parthplc/facebook-hateful-meme-dataset
