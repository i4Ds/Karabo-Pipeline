module load daint-gpu
module load cray-mpich
module load cray-python
python -m venv --system-site-packages slurm_dask_test
source ./slurm_dask_test/bin/activate
pip install dask_mpi