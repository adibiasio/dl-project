#!/bin/bash
#SBATCH --job-name=dl-project
#SBATCH -t 8:00:00              		# Runtime in D-HH:MM
#SBATCH -n 8                          # number of CPU cores
#SBATCH -N 1
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH -C "H100"
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source .venv/bin/activate
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

BASE=/storage/ice1/2/1/dkwon70/dl-project
export HF_HOME=$BASE/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export TMPDIR=$BASE/tmp

mkdir -p $HF_HOME
mkdir -p $TMPDIR

python basic_finetune_blip2.py