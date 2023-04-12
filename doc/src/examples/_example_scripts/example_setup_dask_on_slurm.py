from karabo.util.dask import dask_cleanup, setup_dask_for_slurm

# Setup Dask
client = setup_dask_for_slurm()

# Clean up Dask
dask_cleanup(client)
