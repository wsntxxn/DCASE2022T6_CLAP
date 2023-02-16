#!/bin/bash
#SBATCH --partition=gpu,2080ti
#SBATCH --cpus-per-task=4
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBTACH --mem=20G
#SBATCH --output=slurm_logs/%j.log
#SBATCH --error=slurm_logs/%j.err


sync_file=$(pwd)"/sync"


if [ -e $sync_file ]; then
    rm $sync_file
fi

srun python main_ddp.py train \
    --config configs/cnn14_bertmedium.json \
    --sync_file $sync_file

