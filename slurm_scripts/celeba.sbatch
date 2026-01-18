#!/bin/bash
#SBATCH --job-name=celeba
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02-00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --exclude=babel-v9-28,babel-x5-24
#SBATCH --output=/home/xinyueai/Experiments/flow-maps/log/celeba_%A_%a.out
#SBATCH --error=/home/xinyueai/Experiments/flow-maps/log/celeba_%A_%a.err


## Configuration
PY_DIR=/home/xinyueai/Experiments/flow-maps/py
CONFIG_PATH="configs.celeba64"
OUTPUT_FOLDER="/data/user_data/xinyueai/flow-maps/celeba-lsd"
DATASET_LOCATION="/home/xinyueai/Experiments/data"

# Single experiment: LSD
SLURM_ID=0
METHOD="LSD"
PFMM_TYPE="N/A"
STOPGRAD="convex"

# Print job info
echo "=========================================="
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

echo "CelebA Loss Method Comparison Training"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Method: ${METHOD}"
echo "PFMM Type: ${PFMM_TYPE}"
echo "Stopgrad Type: ${STOPGRAD}"
echo "Sampling: uniform_upper_triangle (half plane)"
echo "Diagonal Fraction: 0.75 (75%)"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_FOLDER}"
echo "=========================================="

# Load modules
module purge

# Hook conda/mamba into the shell and activate env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flowmaps

# Fix library path to prioritize conda environment libraries over system ones
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export WANDB_ENTITY=axy782-carnegie-mellon-university


# Change to project directory
cd $PY_DIR || exit

echo "Starting training..."
echo "Time limit: 48 hours"
echo "Configuration: ${CONFIG_PATH}"
echo "Output folder: ${OUTPUT_FOLDER}"
echo "Dataset: CelebA (64x64)"

# Run training with the specific configuration
python launchers/learn.py \
    --cfg_path "${CONFIG_PATH}" \
    --slurm_id ${SLURM_ID} \
    --dataset_location "${DATASET_LOCATION}" \
    --output_folder "${OUTPUT_FOLDER}"

echo "=========================================="
echo "Job completed. Check the following for results:"
echo "  Output folder: ${OUTPUT_FOLDER}"
echo "  WandB project: fmm2"
echo "=========================================="

exit