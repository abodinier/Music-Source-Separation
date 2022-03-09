#!/bin/sh

<<<<<<< HEAD
#SBATCH --job-name=med_sr          # name of the job UQS37
=======
#SBATCH --job-name=UQS37          # name of the job UQS37
>>>>>>> df5dabdc3f9e7355265c8f958c0987908dd27467
#SBATCH --partition=gpu_p2      # request for allocation on the CPU partition
#SBATCH --ntasks=1                  # number of tasks (a single process here)
#SBATCH --cpus-per-task=2       # number of OpenMP threads
##SBATCH --gres=gpu:2            # number of GPU Please comment it !!!!!!!!!!!!!!!!!!
##SBATCH --hint=nomultithread         # hyperthreading desactive
##SBATCH -C v100-32g
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --time=95:59:59             ##95:59:59            # maximum execution time requested (HH:MM:SS)a
#SBATCH --output=omp%j.out          # name of output file
#SBATCH --error=omp%j.out           # name of error file (here, in common with outp
#SBATCH --gres=gpu:1
##sbatCH --mem-per-cpu=10000
#SBATCH --qos=qos_gpu-t4 # Pour specifier une QoS différente du défaut, vous pouvez au choix : - qos_gpu-t3 (défaut)  20h  96 noeuds- qos_gpu-t4  100h 1 noeud- qos_gpu-dev 2h  4 noeuds
#SBATCH --hint=nomultithread


#SBATCH --account=ldr@gpu


hostname
echo --------------------------------------
echo --------------------------------------
pwd
echo --------------------------------------
echo --------------------------------------

module purge
module load anaconda-py3

conda activate pytorch

CL_SOCKET_IFNAME=eno1 python train_kaituoxu.py --data_dir /gpfsdswork/dataset/MUSDB18/ --ckpdir weights --cfg_path cfg_med_sr.yaml
