#!/bin/bash
#SBATCH --job-name=dl-preprocess-degraded
#SBATCH -t 4:00:00
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# -----------------------
# go to project root (IMPORTANT)
# -----------------------
cd /home/hice1/dkwon70/scratch/dl-project

# -----------------------
# activate environment
# -----------------------
source .venv/bin/activate

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# -----------------------
# HF / temp setup
# -----------------------
export HF_HOME=/storage/ice1/2/1/dkwon70/dl-project/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# keep temp local to job (fine on scratch)
export TMPDIR=/home/hice1/dkwon70/scratch/dl-project/tmp

mkdir -p $HF_HOME
mkdir -p $TMPDIR
mkdir -p logs

# -----------------------
# select level
# -----------------------
LEVELS=("medium" "heavy")
LEVEL=${LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "Running preprocessing for $LEVEL..."

python blip2/blip2_preprocessing_degraded.py --level $LEVEL

echo "Finished $LEVEL"