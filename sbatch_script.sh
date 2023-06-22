#!/bin/bash -l
#SBATCH --job-name="slurm_dask_timing"
#SBATCH --account="sk05"
#SBATCH --nodes=3
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --partition=debug

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

# Load module for dask-mpi
source ./slurm_dask_test/bin/activate
module load cray-mpich

# Define the folder path for the scheduler file
scheduler_file_folder=".karabo_dask/scheduler_files"

# Create the folder if it doesn't exist
mkdir -p "$scheduler_file_folder"

# Generate a unique ID using the job ID
unique_id="$SLURM_JOB_ID"

# Set the environment variable with the scheduler file path
export KARABO_DASK_SCHEDULER_FILE="$scheduler_file_folder/scheduler_file_${unique_id}.json"

# Obtain the number of nodes from SLURM environment variables
num_nodes=$SLURM_JOB_NUM_NODES

# Calculate the number of processes based on the number of nodes
num_processes=$((num_nodes * SLURM_NTASKS_PER_NODE))

# Start the Dask cluster using dask-mpi and provide the path for the scheduler file
srun -n $num_nodes dask-mpi --scheduler-file "$KARABO_DASK_SCHEDULER_FILE"

# Run the script.
conda activate karabo_dev_env
srun python3 karabo/simulation/line_emission.py

# Cleanup on exit
trap "rm -f $KARABO_DASK_SCHEDULER_FILE" EXIT