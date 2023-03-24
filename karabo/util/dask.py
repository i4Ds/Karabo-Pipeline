import copy
import os
import time
from datetime import datetime
from subprocess import call
from typing import Callable, List

import dask
import psutil
from dask import delayed
from distributed import Client, LocalCluster

client = None


def get_global_client(
    min_ram_gb_per_worker: int = 20, threads_per_worker: int = 1
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

        # we pass telescope as we want to use the same in every simulation,
        # and so we only need to setup one.
        # pass flux so we can use a different one in each iteration.
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
    :return: list of delayed objects. Need to be calculated with dask.compute() later
    """
    results = []
    for i in range(0, n):
        res = delayed(function)(*[copy.deepcopy(arg) for arg in args])
        results.append(res)
    return dask.compute(*results)


def parallel_for_each(arr: List[any], function: Callable, *args):
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
    return dask.compute(*results)


def setup_dask_for_slurm():
    # Detect if we are on a slurm cluster
    if "SLURM_JOB_ID" not in os.environ or os.getenv("SLURM_JOB_NUM_NODES") == "1":
        print("Not on a SLURM cluster or only 1 node. Not setting up dask.")
        return None
    else:
        if is_first_node():
            # Remove old scheduler file
            try:
                os.remove("scheduler.txt")
            except FileNotFoundError:
                pass

            # Create client and scheduler
            cluster = LocalCluster(ip=get_lowest_node_name())
            client = Client(cluster)

            # Write the scheduler address to a file
            with open("scheduler.txt", "w") as f:
                f.write(cluster.scheduler_address)

            print(
                f'Main Node. Name = {os.getenv("SLURMD_NODENAME")}. Client = {client}'
            )

            while (
                len(client.scheduler_info()["workers"])
                < int(os.getenv("SLURM_JOB_NUM_NODES")) + 1
            ):
                print(
                    f"Waiting for all workers to connect. Current number of workers: "
                    f"{len(client.scheduler_info()['workers'])}. "
                    f"NNodes: {os.getenv('SLURM_JOB_NUM_NODES')}"
                )
                time.sleep(3)

            # Print the number of workers
            print(f'Number of workers: {len(client.scheduler_info()["workers"])}')
            return client

        else:
            # Sleep first to make sure no old scheduler file is read
            time.sleep(5)

            # Read the scheduler address from the file
            scheduler_address = None
            timeout_time = datetime.now().timestamp() + 60
            while scheduler_address is None:
                try:
                    with open("scheduler.txt", "r") as f:
                        scheduler_address = f.read()
                except FileNotFoundError:
                    time.sleep(1)
                if datetime.now().timestamp() > timeout_time:
                    raise TimeoutError(
                        "Timeout while waiting for scheduler file to appear."
                    )
            print(
                f"Worker Node. Name = {os.getenv('SLURMD_NODENAME')}."
                f"Scheduler Address = {scheduler_address}"
            )
            call(["dask", "worker", scheduler_address])


def get_min_max_of_node_id():
    """
    Returns the min max from SLURM_JOB_NODELIST. Can handle if it runs only on two
    nodes (separated with a comma) of if it runs on more than two nodes (separated with a dash).
    """
    node_list = os.getenv("SLURM_JOB_NODELIST").split("[")[1].split("]")[0]
    if "," in node_list:
        return int(node_list.split(",")[0]), int(node_list.split(",")[1])
    else:
        return int(node_list.split("-")[0]), int(node_list.split("-")[1])


def get_lowest_node_id():
    return get_min_max_of_node_id()[0]


def get_lowest_node_name():
    return os.getenv("SLURM_JOB_NODELIST").split("[")[0] + str(get_lowest_node_id())


def create_list_of_node_names():
    return [
        os.getenv("SLURM_JOB_NODELIST").split("[")[0] + str(i)
        for i in range(get_min_max_of_node_id()[0], get_min_max_of_node_id()[1] + 1)
    ]


def get_node_id():
    len_id = len(str(get_lowest_node_id()))
    return int(os.getenv("SLURMD_NODENAME")[-len_id:])


def is_first_node():
    return get_node_id() == get_lowest_node_id()


def get_current_time():
    return time.strftime("%H:%M:%S", time.localtime())
