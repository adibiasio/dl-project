#!/bin/bash
#SBATCH --job-name=img-degrade
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# ----------------------------
# Environment setup
# ----------------------------
source .venv/bin/activate

BASE=/storage/ice1/2/1/dkwon70/dl-project

export HF_HOME=$BASE/hf-cache
export TMPDIR=$BASE/tmp

mkdir -p logs
mkdir -p $TMPDIR

# ----------------------------
# Run degradation script
# ----------------------------

echo "Starting degradation job..."

python generate_degraded_images.py --level medium

echo "Finished medium level"

python generate_degraded_images.py --level heavy

echo "Finished heavy level"

echo "All degradation jobs complete"