#!/bin/bash

#SBATCH --job-name=uORF
#SBATCH --ntasks=1
#SBATCH --time=24:00:00

# Number of gpus. Available nodes: dev_gpu_4, gpu_4, gpu_8
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu_8

# Output files
#SBATCH --output=../out/R-%x.%j.out
#SBATCH --error=../out/R-%x.%j.err

# 216:00:00
# 48:00:00
# 00:30:00

echo 'Activate conda environment'
source ~/.bashrc
conda activate uorf-3090

echo 'Start training'
exec custom/train_room_diverse.sh