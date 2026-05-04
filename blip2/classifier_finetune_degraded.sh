#!/bin/bash
#SBATCH --job-name=dl-project
#SBATCH --array=0-2
#SBATCH -t 8:00:00              		# Runtime in D-HH:MM
#SBATCH -n 1                          # number of CPU cores
#SBATCH -N 1
#SBATCH --mem=96G
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH -C "H100"
#SBATCH --output=logs/%x-classifier-%j.out
#SBATCH --error=logs/%x-classifier-%j.err

cd /home/hice1/dkwon70/scratch/dl-project
source /home/hice1/dkwon70/scratch/dl-project/.venv/bin/activate
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

LEVELS=("clean" "medium" "heavy")
LEVEL=${LEVELS[$SLURM_ARRAY_TASK_ID]}

echo "Running training for $LEVEL..."

cd /home/hice1/dkwon70/scratch/dl-project
source /home/hice1/dkwon70/scratch/dl-project/.venv/bin/activate
/home/hice1/dkwon70/scratch/dl-project/.venv/bin/python \
/home/hice1/dkwon70/scratch/dl-project/blip2/classifier_finetune_blip2_degraded.py \
--level $LEVEL