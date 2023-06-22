import os
import time
from datetime import datetime
from random import random

from dask.distributed import Client
from dask_mpi import initialize
from mpi4py import MPI


def get_current_time():
    return time.strftime("%H:%M:%S", time.localtime())

def slow_is_prime(num):
    print(f'Calculating {num} on node {os.getenv("SLURMD_NODENAME")} at {get_current_time()}')
    time.sleep(1)
    if num == 1:
        return False
    for test_factor in range(2, num // 2):
        if num % test_factor == 0:
            return False

    return True

if __name__ == '__main__':
    num_threads = int(os.environ.get(
        'SLURM_CPUS_PER_TASK',
        os.environ.get('OMP_NUM_THREADS', 1)
    ))
    initialize(nthreads=num_threads, comm=MPI.COMM_WORLD)
    client = Client()
    start_time = datetime.now()
    # Print the number of workers
    print(f'Number of workers: {len(client.scheduler_info()["workers"])}')
    # Prime between two numbers
    low = 1
    high = 100
    num_primes = sum(
        client.gather(
            client.map(slow_is_prime, range(low, high))
        )
    )
    end_time = datetime.now()
    print(f'{num_primes} primes between {low} and {high}. Time taken: {end_time - start_time}]. NNodes: {os.getenv("SLURM_JOB_NUM_NODES")}, NThreads: {num_threads}')