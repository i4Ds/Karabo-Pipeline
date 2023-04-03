#!/bin/bash -l
#SBATCH --job-name="slurm_dask_timing"
#SBATCH --account="sk05"
#SBATCH --nodes=2
#SBATCH --time=00:05:00
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --partition=debug

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
conda activate karabo_dev_env
srun python3 cscs_rascil_dask.py