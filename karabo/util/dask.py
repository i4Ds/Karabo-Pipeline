from __future__ import annotations

import atexit
import json
import os
import time
from subprocess import Popen
from typing import Optional, Tuple

import psutil
from dask.distributed import Client, LocalCluster

from karabo.warning import KaraboWarning

DASK_INFO_ADDRESS = os.path.join(".karabo_dask", "dask_info.json")
os.makedirs(os.path.dirname(DASK_INFO_ADDRESS), exist_ok=True)


class DaskHandler:
    """
    A class for managing a Dask client.

    Attributes
    ----------
    dask_client: Optional[Client]
        The Dask client object. If None, a new client will be created.
    n_workers_scheduler_node : int
        The number of workers to start on the scheduler node.
    threads_per_worker : int
        The number of threads to allocate per worker.
    min_ram_per_worker : Optional[float]
        The minimum RAM to allocate per worker in GB.

    Methods
    -------
    get_dask_client() -> Client:
        Returns a Dask client object. If the client does not exist, and
        the current node is a SLURM node and there are more than 1 node,
        a Dask client will be created.

    """

    dask_client: Optional[Client] = None
    n_workers_scheduler_node = 1
    threads_per_worker = 1
    min_ram_per_worker = None
    use_dask = None
    TIMEOUT = 60

    @staticmethod
    def get_dask_client() -> Client:
        if DaskHandler.dask_client is None:
            if is_on_slurm_cluster() and get_number_of_nodes() > 1:
                DaskHandler.dask_client = setup_dask_for_slurm(
                    DaskHandler.n_workers_scheduler_node,
                    DaskHandler.threads_per_worker,
                    DaskHandler.min_ram_per_worker,
                )
            else:
                DaskHandler.dask_client = get_local_dask_client(
                    DaskHandler.min_ram_per_worker, DaskHandler.threads_per_worker
                )

        atexit.register(dask_cleanup, DaskHandler.dask_client)
        return DaskHandler.dask_client

    def should_dask_be_used(override: Optional[bool] = None) -> bool:
        if override is not None:
            return override
        elif DaskHandler.use_dask is not None:
            return DaskHandler.use_dask
        elif DaskHandler.dask_client is not None:
            return True
        elif is_on_slurm_cluster() and get_number_of_nodes() > 1:
            return True
        else:
            return False


def get_local_dask_client(
    min_ram_gb_per_worker: int, threads_per_worker: int
) -> Client:
    # Calculate number of workers per node
    n_workers = calculate_number_of_workers_per_node(min_ram_gb_per_worker)
    client = Client(
        LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    )
    return client


def dask_cleanup(client: Client) -> None:
    # Remove the scheduler file if somehow it was not removed
    if os.path.exists(DASK_INFO_ADDRESS):
        os.remove(DASK_INFO_ADDRESS)

    if client is not None:
        client.close()
        client.shutdown()


def prepare_slurm_nodes_for_dask() -> None:
    # Detect if we are on a slurm cluster
    if not is_on_slurm_cluster() or get_number_of_nodes() <= 1:
        print("Not on a SLURM cluster or only 1 node. Not setting up dask.")
        DaskHandler.use_dask = False
        return
    else:
        print("Detected SLURM cluster. Setting up dask.")

    # Check if we are on the first node
    if is_first_node():
        # Remove old scheduler file
        if os.path.exists(DASK_INFO_ADDRESS):
            os.remove(DASK_INFO_ADDRESS)

    else:
        # Wait some time to make sure the scheduler file is new
        time.sleep(10)

        # Wait until dask info file is created
        while not os.path.exists(DASK_INFO_ADDRESS):
            time.sleep(1)

        # Load dask info file
        with open(DASK_INFO_ADDRESS, "r") as f:
            dask_info = json.load(f)

        _ = Popen(
            [
                "dask",
                "worker",
                dask_info["scheduler_address"],
                "--nthreads",
                str(dask_info["n_threads_per_worker"]),
                "--nworkers",
                str(dask_info["n_workers_per_node"]),
            ]
        )

        while True:
            # Wait some time
            time.sleep(5)


def calculate_number_of_workers_per_node(min_ram_gb_per_worker: int) -> int:
    if min_ram_gb_per_worker is None:
        return 1
    # Calculate number of workers per node
    ram = psutil.virtual_memory().available / 1024 / 1024
    n_workers_per_node = int(ram / (min_ram_gb_per_worker))
    if ram < min_ram_gb_per_worker:
        KaraboWarning(
            f"Only {ram} GB of RAM available. Requested at least "
            f"{min_ram_gb_per_worker} GB. Setting number of "
            f"workers per node to 1."
        )
        n_workers_per_node = 1

    if n_workers_per_node > psutil.cpu_count():
        KaraboWarning(
            f"Only {psutil.cpu_count()} CPUs available. Requested "
            f"{n_workers_per_node} workers per node. Setting number of "
            f"workers per node to {psutil.cpu_count()}."
        )
        n_workers_per_node = psutil.cpu_count()

    return n_workers_per_node


def setup_dask_for_slurm(
    n_workers_scheduler_node: int,
    n_threads_per_worker: int,
    min_ram_gb_per_worker: int,
) -> Client:
    if is_first_node():
        # Create client and scheduler
        cluster = LocalCluster(
            ip=get_lowest_node_name(),
            n_workers=n_workers_scheduler_node,
            threads_per_worker=n_threads_per_worker,
        )
        dask_client = Client(cluster)

        # Calculate number of workers per node
        n_workers_per_node = calculate_number_of_workers_per_node(min_ram_gb_per_worker)

        # Create dictionary with the information
        dask_info = {
            "scheduler_address": cluster.scheduler_address,
            "n_workers_per_node": n_workers_per_node,
            "n_threads_per_worker": n_threads_per_worker,
        }

        # Write scheduler file
        with open(DASK_INFO_ADDRESS, "w") as f:
            json.dump(dask_info, f)

        # Wait until all workers are connected
        n_workers_requested = (
            get_number_of_nodes() - 1
        ) * n_workers_per_node + n_workers_scheduler_node

        start = time.time()
        while len(dask_client.scheduler_info()["workers"]) != n_workers_requested:
            print(
                f"Waiting for all workers to connect. Currently "
                f"{len(dask_client.scheduler_info()['workers'])} "
                f"workers connected of {n_workers_requested} requested."
            )
            time.sleep(3)  # To avoid spamming the scheduler
            if time.time() - start > DaskHandler.TIMEOUT:
                raise Exception(
                    "Timeout while waiting for all workers to connect. "
                    "Something went wrong."
                )

        print(f"All {len(dask_client.scheduler_info()['workers'])} workers connected!")
        atexit.register(dask_cleanup, dask_client)

        # Removing file
        if os.path.exists(DASK_INFO_ADDRESS):
            os.remove(DASK_INFO_ADDRESS)
        return dask_client

    else:
        raise Exception("This function should only be reached on the first node.")


def get_min_max_of_node_id() -> Tuple[int, int]:
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


def get_lowest_node_id() -> int:
    if get_number_of_nodes() == 1:
        return get_node_id()
    else:
        return get_min_max_of_node_id()[0]


def get_base_string_node_list() -> str:
    return os.getenv("SLURM_JOB_NODELIST").split("[")[0]


def get_lowest_node_name() -> str:
    return get_base_string_node_list() + str(get_lowest_node_id())


def get_number_of_nodes() -> int:
    n_nodes = os.getenv("SLURM_JOB_NUM_NODES")
    return int(n_nodes)


def get_node_id() -> int:
    len_id = len(str(get_lowest_node_id()))
    return int(os.getenv("SLURMD_NODENAME")[-len_id:])


def is_first_node() -> bool:
    return get_node_id() == get_lowest_node_id()


def is_on_slurm_cluster() -> bool:
    return "SLURM_JOB_ID" in os.environ
