#!/bin/bash
#SBATCH -p iaensta
#SBATCH -t 47:10:00 # time limit max 48:00:00
#SBATCH -c 4 # number of cores
#SBATCH --gres=gpu:1 # number of gpus required max 4 (will be limited to 1)
#SBATCH -o /home/group7IA/thomas/results.out #where to write sys.out/print


dir=$PWD
source ~/.bashrc
echo "Starting"
eval "$(conda shell.bash hook)"
conda activate VAE
cd $dir
srun python main.py -train=True