import os
import time

from mpi4py import MPI

from karabo.util.dask import DaskHandler


def sleep_and_print_info(sleep_time):
    """
    Sleeps for a given time and then prints out information about the node.

    Parameters:
    - sleep_time: Time in seconds to sleep.
    """
    time.sleep(sleep_time)
    node_id = os.environ.get("SLURM_NODEID", "Unknown Node ID")
    job_id = os.environ.get("SLURM_JOB_ID", "Unknown Job ID")
    return f"SLURM Node ID: {node_id}, SLURM Job ID: {job_id}, Slept for: {sleep_time} seconds"


if __name__ == "__main__":
    DaskHandler.setup()
    print("MPI Rank: ", MPI.COMM_WORLD.Get_rank())
    print("MPI Size: ", MPI.COMM_WORLD.Get_size())
    # Initialize Dask client
    client = DaskHandler.get_dask_client()

    # Define the sleep time
    sleep_time = 5  # seconds

    # Retrieve the number of workers
    num_workers = len(client.scheduler_info()["workers"])

    # Create multiple Dask jobs, one for each worker
    futures = client.map(sleep_and_print_info, [sleep_time] * num_workers)

    # Wait for all jobs to complete and gather results
    results = client.gather(futures)

    # Print the results
    for result in results:
        print(result)
