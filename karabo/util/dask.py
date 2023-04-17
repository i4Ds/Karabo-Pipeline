from __future__ import annotations

import atexit
import os
import time
from subprocess import call
from typing import Optional

from dask.distributed import Client, LocalCluster

SCHEDULER_ADDRESS = "scheduler_address.json"

class DaskHandler():
    dask_client: Optional[Client] = None
    _n_workers_scheduler_node: int = 1

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_dask_client(n_workers_scheduler_node: int = 1) -> Client:
        if DaskHandler.dask_client is None:
            DaskHandler.dask_client = setup_dask_for_slurm(n_workers_scheduler_node=n_workers_scheduler_node)
            DaskHandler._n_workers_scheduler_node = n_workers_scheduler_node
        elif DaskHandler._n_workers_scheduler_node != n_workers_scheduler_node:
            raise Exception("Dask client already created with different number of workers.")
        return DaskHandler.dask_client

def dask_cleanup(client: Client):
    # Remove the scheduler file
    if os.path.exists(SCHEDULER_ADDRESS):
        os.remove(SCHEDULER_ADDRESS)

    if client is not None:
        client.close()
        client.shutdown()

def prepare_slurm_nodes_for_dask():
    # Detect if we are on a slurm cluster
    if not is_on_slurm_cluster() or os.getenv("SLURM_JOB_NUM_NODES") == "1":
        print("Not on a SLURM cluster or only 1 node. Not setting up dask.")
        return 
    
    # Check if we are on the first node
    if is_first_node():
        # Remove old scheduler file
        if os.path.exists(SCHEDULER_ADDRESS):
            os.remove(SCHEDULER_ADDRESS)
    
    else:
        # Wait some time to make sure the scheduler file is new
        time.sleep(10)

        # Wait until scheduler file is created
        while not os.path.exists(SCHEDULER_ADDRESS):
            print("Waiting for scheduler file to be created.")
            time.sleep(1)

        # Read scheduler file
        with open(SCHEDULER_ADDRESS, "r") as f:
            scheduler_address = f.read()

        # Create client
        call(["dask", "worker", scheduler_address])

        # Run until client is closed
        while True:
            time.sleep(5)

def setup_dask_for_slurm(n_workers_scheduler_node: int = 1):
    if is_first_node():
        # Create client and scheduler
        cluster = LocalCluster(
            ip=get_lowest_node_name(), n_workers=n_workers_scheduler_node
        )
        dask_client = Client(cluster)

        # Write scheduler file
        with open(SCHEDULER_ADDRESS, "w") as f:
            f.write(cluster.scheduler_address)

        # Wait until all workers are connected
        n_workers_requested = get_number_of_nodes() - 1 + n_workers_scheduler_node
        while len(dask_client.scheduler_info()["workers"]) < n_workers_requested:
            print(
                f"Waiting for all workers to connect. Currently "
                f"{len(dask_client.scheduler_info()['workers'])} "
                f"workers connected of {n_workers_requested} requested."
            )
            time.sleep(1)

        print(f"All {len(dask_client.scheduler_info()['workers'])} workers connected!")
        atexit.register(dask_cleanup, dask_client)
        return dask_client

    else:
        raise Exception("This function should only be reached on the first node.")


def get_min_max_of_node_id():
    """
    Returns the min max from SLURM_JOB_NODELIST.
    Works if it's run only on two nodes (separated with a comma)
    of if it runs on more than two nodes (separated with a dash).
    """
    node_list = os.getenv("SLURM_JOB_NODELIST").split("[")[1].split("]")[0]
    if "," in node_list:
        return int(node_list.split(",")[0]), int(node_list.split(",")[1])
    else:
        return int(node_list.split("-")[0]), int(node_list.split("-")[1])


def get_lowest_node_id():
    return get_min_max_of_node_id()[0]


def get_base_string_node_list():
    return os.getenv("SLURM_JOB_NODELIST").split("[")[0]


def get_lowest_node_name():
    return get_base_string_node_list() + str(get_lowest_node_id())


def get_number_of_nodes():
    return get_min_max_of_node_id()[1] - get_min_max_of_node_id()[0] + 1


def create_node_list_except_first():
    """
    Returns a list of all nodes except the first one to pass to SLURM
    Example: node[2-4] if there are 4 nodes or node[2] if there are 2 nodes
    """
    min_node, max_node = get_min_max_of_node_id()
    if get_number_of_nodes() == 2:
        return get_base_string_node_list() + "[" + str(min_node + 1) + "]"

    return (
        get_base_string_node_list()
        + "["
        + str(min_node + 1)
        + "-"
        + str(max_node)
        + "]"
    )


def get_node_id():
    len_id = len(str(get_lowest_node_id()))
    return int(os.getenv("SLURMD_NODENAME")[-len_id:])


def is_first_node():
    return get_node_id() == get_lowest_node_id()


def is_on_slurm_cluster():
    return "SLURM_JOB_ID" in os.environ
