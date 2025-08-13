#!/bin/sh
#SBATCH --time=3:0:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBTACH --tmp=200G
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=karabo
#SBATCH --out=output/benchmark.out
#SBATCH --err=output/benchmark.err
#SBATCH --array=50,100,200,400

# needed for matplotlib
mkdir $TMPDIR/.config
mkdir $TMPDIR/.cache

cd $SCRATCH/karabo-benchmark

# starting Karabo from Singularity container
apptainer exec \
--writable-tmpfs \
--nv \
--bind $TMPDIR/.cache \
--bind $TMPDIR/.config \
--bind $SCRATCH:/workspace \
../karabo_0.31.0.sif bash -c "cd /workspace/karabo-benchmark; python karabo_benchmark.py $SLURM_ARRAY_TASK_ID"
