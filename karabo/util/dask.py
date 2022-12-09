import copy
from typing import Callable, List

from dask.delayed import delayed
from dask.base import compute
from distributed import Client
from distributed.deploy.local import LocalCluster
import psutil

client = None


def get_global_client(
    min_ram_gb_per_worker: int = 2, threads_per_worker: int = 1
) -> Client:
    global client
    if client is None:
        client = get_local_dask_client(min_ram_gb_per_worker, threads_per_worker)
    print(f"Client Dashboard Address: {client.dashboard_link}")
    return client


def get_local_dask_client(min_ram_gb_per_worker, threads_per_worker) -> Client:
    global client
    if client is not None:
        return client
    min_ram_gb_per_worker *= 1024
    cpus = psutil.cpu_count()
    ram = psutil.virtual_memory().total / 1024 / 1024
    if ram / cpus >= min_ram_gb_per_worker:
        client = Client(
            LocalCluster(n_workers=cpus, threads_per_worker=threads_per_worker)
        )
    else:
        workers = cpus
        while ram / workers < min_ram_gb_per_worker:
            workers -= 1

        client = Client(
            LocalCluster(n_workers=workers, threads_per_worker=threads_per_worker)
        )
    return client


def parallel_for(n: int, function: Callable, *args):
    """
    Execute a function ``n`` times in parallel with DASK.

    For example creating many simulations at once for running in parallel::

        # we pass telescope as we want to use the same in every simulation, and so we only need to
        # setup one. pass flux so we can use a different one in each iteration.
        def my_simulation_code(telescope, flux):
            # setup simulation settings and observation settings
            simulation = InterferometerSimulation(...)
            observation = Observation(...)
            sky = SkyModel(np.array([[20, -30, flux]]))
            sim_result = simulation.run_simulation(telescope, sky, observation)
            return sim_result

        flux = 0.001
        telescope = get_ASKAP_telescope()
        results = parallel_for(10, # how many iterations the loop will do.
            my_simluation_code, # code to execute 10 times
            telescope, # param 1 for passed function
            flux + flux + 0.001) # param 2 for passed function
        # Start compute to actually start the computation.
        results = compute(*results)

    :param n: number of iterations
    :param function: function to execute n times
    :param args: arguments that will be passed to the passed function
    :return: list of delayed objects. that need to be calculated with dask.compute() later
    """
    results = []
    for i in range(0, n):
        res = delayed(function)(*[copy.deepcopy(arg) for arg in args])
        results.append(res)
    return compute(*results)


def parallel_for_each(arr: List, function: Callable, *args):
    """
    :param arr:
    :param function:
    :param args:
    :return:
    """
    results = []
    for value in arr:
        res = delayed(function)(value, *[copy.deepcopy(arg) for arg in args])
        results.append(res)
    return compute(*results)
