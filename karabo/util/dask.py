import copy
import os
import sys
import time
from datetime import datetime
from subprocess import call
from typing import Callable, List

import dask
import psutil
from dask import delayed
from distributed import Client, LocalCluster


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

def setup_dask_for_slurm(number_of_workers_on_scheduler_node: int = 1):
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
            cluster = LocalCluster(
                ip=get_lowest_node_name(), n_workers=number_of_workers_on_scheduler_node
            )
            client = Client(cluster)

            # Write the scheduler address to a file
            with open("scheduler.txt", "w") as f:
                f.write(cluster.scheduler_address)

            print(f'Main Node. Name = {os.getenv("SLURMD_NODENAME")}. Client = {client}')
                
            while len(client.scheduler_info()['workers']) != int(os.getenv('SLURM_JOB_NUM_NODES')):
                print(f'Waiting for all workers to connect. Current number of workers: {len(client.scheduler_info()["workers"])}. NNodes: {os.getenv("SLURM_JOB_NUM_NODES")}')
                # Print IP, Adress and Name of all workers
                for worker in client.scheduler_info()['workers']:
                    print(f'Worker: {worker}. IP: {client.scheduler_info()["workers"][worker]["host"]}. Node Name: {client.scheduler_info()["workers"][worker]["name"]}')
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
            while scheduler_address is None and datetime.now().timestamp() < timeout_time:
                try:
                    with open("scheduler.txt", "r") as f:
                        scheduler_address = f.read()
                except FileNotFoundError:
                    time.sleep(1)
            print(f'Worker Node. Name = {os.getenv("SLURMD_NODENAME")}. Scheduler Address = {scheduler_address}')
            call(['dask', 'worker', scheduler_address])
            sys.exit(1)



def get_min_max_of_node_id():
    """
    Returns the min max from SLURM_JOB_NODELIST. Can handle if it runs only on two nodes (separated with a comma) 
    of if it runs on more than two nodes (separated with a dash).
    """
    node_list = os.getenv('SLURM_JOB_NODELIST').split('[')[1].split(']')[0]
    if ',' in node_list:
        return int(node_list.split(',')[0]), int(node_list.split(',')[1])
    else:
        return int(node_list.split('-')[0]), int(node_list.split('-')[1])

    
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
