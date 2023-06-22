#!/bin/bash -l
#SBATCH --job-name="slurm_dask_test"
#SBATCH --account="sk05"
#SBATCH --time=00:05:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
source slurm_dask_test/bin/activate
module load cray-mpich
srun python3 test_dask_mpi.py