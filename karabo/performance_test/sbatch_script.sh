#!/bin/bash -l
#SBATCH --job-name="slurm_dask_timing"
#SBATCH --account="sk05"
#SBATCH --nodes=4
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --partition=normal

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
conda activate karabo_dev_env
srun python3 time_karabo_parallelization_by_channel.py
