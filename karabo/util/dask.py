from __future__ import annotations

import asyncio
import atexit
import json
import os
import time
from typing import Any, Callable, Optional, Tuple

import psutil
from dask.distributed import Client, LocalCluster, Worker

from karabo.error import KaraboDaskError
from karabo.util._types import IntFloat
from karabo.util.data_util import extract_chars_from_string, extract_digit_from_string
from karabo.warning import KaraboWarning

DASK_INFO_FOLDER = ".karabo_dask"
DASK_INFO_FILE = "dask_info.json"

##
if "SLURM_JOB_ID" in os.environ:
    DASK_INFO_FOLDER = os.path.join(DASK_INFO_FOLDER, str(os.environ["SLURM_JOB_ID"]))
os.makedirs(DASK_INFO_FOLDER, exist_ok=True)
DASK_INFO_ADDRESS = os.path.join(DASK_INFO_FOLDER, DASK_INFO_FILE)


class DaskHandler:
    """
    A class for managing a Dask client.

    Attributes
    ----------
    dask_client: Optional[Client]
        The Dask client object. If None, a new client will be created.
    n_workers_scheduler_node : int
        The number of workers to start on the scheduler node.
    min_gb_ram_per_worker : Optional[float]
        The minimum RAM to allocate per worker in GB.

    Methods
    -------
    get_dask_client() -> Client:
        Returns a Dask client object. If the client does not exist, and
        the current node is a SLURM node and there are more than 1 node,
        a Dask client will be created.

    """

    dask_client: Optional[Client] = None
    n_workers_scheduler_node: int = 1
    min_gb_ram_per_worker: Optional[int] = None
    use_dask: Optional[bool] = None
    TIMEOUT: int = 60

    @staticmethod
    def get_dask_client() -> Client:
        if DaskHandler.dask_client is None:
            if is_on_slurm_cluster() and get_number_of_nodes() > 1:
                DaskHandler.dask_client = setup_dask_for_slurm(
                    DaskHandler.n_workers_scheduler_node,
                    DaskHandler.min_gb_ram_per_worker,
                )
            else:
                DaskHandler.dask_client = get_local_dask_client(
                    DaskHandler.min_gb_ram_per_worker
                )
            # Write the dashboard link to a file
            with open("karabo-dask-dashboard.txt", "w") as f:
                f.write(DaskHandler.dask_client.dashboard_link)
            # Register cleanup function
            print(f"Dashboard link: {DaskHandler.dask_client.dashboard_link}")
            atexit.register(dask_cleanup, DaskHandler.dask_client)
        return DaskHandler.dask_client

    @staticmethod
    def should_dask_be_used(override: Optional[bool] = None) -> bool:
        if override is not None:
            return override
        elif DaskHandler.use_dask is not None:
            return DaskHandler.use_dask
        elif DaskHandler.dask_client is not None:
            return True
        elif is_on_slurm_cluster():
            return True
        else:
            return False


def dask_cleanup(client: Client) -> None:
    # Remove the scheduler file if somehow it was not removed
    if os.path.exists(DASK_INFO_ADDRESS):
        os.remove(DASK_INFO_ADDRESS)

    # Remove the dashboard file if somehow it was not removed
    if os.path.exists("karabo-dask-dashboard.txt"):
        os.remove("karabo-dask-dashboard.txt")

    if client is not None:
        client.close()
        client.shutdown()


def prepare_slurm_nodes_for_dask() -> None:
    # Detect if we are on a slurm cluster
    if not is_on_slurm_cluster() or get_number_of_nodes() <= 1:
        DaskHandler.use_dask = False
        return
    else:
        print("Detected SLURM cluster. Setting up dask.")

    # Check if we are on the first node
    if is_first_node():
        pass

    else:
        print("I am on a node! I need to start a dask worker.")
        print(f"My Node ID: {get_node_id()}")
        # Wait some time to make sure the scheduler file is new
        time.sleep(10)

        # Wait until dask info file is created
        while not os.path.exists(DASK_INFO_ADDRESS):
            time.sleep(1)

        # Load dask info file
        with open(DASK_INFO_ADDRESS, "r") as f:
            dask_info = json.load(f)

        print("I am on a node! I need to start a dask worker.")
        print(f"My Node ID: {get_node_id()}")
        # Wait some time to make sure the scheduler file is new
        time.sleep(10)

        # Wait until dask info file is created
        while not os.path.exists(DASK_INFO_ADDRESS):
            time.sleep(1)

        # Load dask info file
        with open(DASK_INFO_ADDRESS, "r") as f:
            dask_info = json.load(f)

        async def start_worker(scheduler_address):
            worker = await Worker(scheduler_address)
            await worker.finished()

        scheduler_address = dask_info["scheduler_address"]
        # Number of workers you want to start
        n_workers = dask_info["n_workers_per_node"]

        # Start workers
        for _ in range(n_workers):
            asyncio.run(start_worker(scheduler_address))

        # Wait for the script to finish and for the
        # kill signal to be sent
        while True:
            time.sleep(10)


def calculate_number_of_workers_per_node(
    min_ram_gb_per_worker: Optional[IntFloat],
) -> int:
    if min_ram_gb_per_worker is None:
        return 1
    # Calculate number of workers per node
    ram = psutil.virtual_memory().available / 1e9  # GB
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


def get_local_dask_client(
    min_ram_gb_per_worker: Optional[IntFloat],
) -> Client:
    # Calculate number of workers per node
    n_workers = calculate_number_of_workers_per_node(min_ram_gb_per_worker)
    print(f"Node name: {get_node_name()}")
    client = Client(
        LocalCluster(
            ip=get_node_name() if is_on_slurm_cluster() else None,
            n_workers=n_workers,
        )
    )
    return client


def setup_dask_for_slurm(
    n_workers_scheduler_node: int,
    min_ram_gb_per_worker: Optional[IntFloat],
) -> Client:
    if is_first_node():
        # Create client and scheduler
        print(f"First node. Name = {get_lowest_node_name()}")
        cluster = LocalCluster(ip=get_node_name(), n_workers=n_workers_scheduler_node)
        dask_client = Client(cluster)

        # Calculate number of workers per node
        n_workers_per_node = calculate_number_of_workers_per_node(min_ram_gb_per_worker)

        # Create dictionary with the information
        dask_info = {
            "scheduler_address": cluster.scheduler_address,
            "n_workers_per_node": n_workers_per_node,
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
                raise KaraboDaskError(
                    "Timeout while waiting for all workers to connect. "
                    "Something went wrong."
                )

        print(f"All {len(dask_client.scheduler_info()['workers'])} workers connected!")
        return dask_client

    else:
        raise KaraboDaskError("This function should only be reached on the first node.")


def get_min_max_of_node_id() -> Tuple[str, str]:
    """
    Returns the min max from SLURM_JOB_NODELIST.
    Works if it's run only on two nodes (separated with a comma)
    of if it runs on more than two nodes (separated with a dash).
    """
    slurm_job_nodelist = check_env_var(
        var="SLURM_JOB_NODELIST", fun=get_min_max_of_node_id
    )
    if get_number_of_nodes() == 1:
        # Node name will be something like "psanagpu115"
        min_max = extract_digit_from_string(slurm_job_nodelist)
        print(f"Node min_max: {min_max}")
        return min_max, min_max

    node_list = slurm_job_nodelist.split("[")[1].split("]")[0]
    # If there is a comma, it means that there are only two nodes
    # Example: psanagpu115,psanagpu116
    # If there is a dash, it means that there are more than two nodes
    # Example: psanagpu115-psanagpu117
    if "," in node_list:
        return node_list.split(",")[0], node_list.split(",")[1]
    else:
        return node_list.split("-")[0], node_list.split("-")[1]


def get_lowest_node_id() -> int:
    return get_min_max_of_node_id()[0]


def get_base_string_node_list() -> str:
    slurm_job_nodelist = check_env_var(
        var="SLURM_JOB_NODELIST", fun=get_base_string_node_list
    )
    if get_number_of_nodes() == 1:
        return extract_chars_from_string(slurm_job_nodelist)
    else:
        return slurm_job_nodelist.split("[")[0]


def get_lowest_node_name() -> str:
    return get_base_string_node_list() + str(get_lowest_node_id())


def get_number_of_nodes() -> int:
    n_nodes = check_env_var(var="SLURM_JOB_NUM_NODES", fun=get_number_of_nodes)
    return int(n_nodes)


def get_node_id() -> str:
    # Attention, often the node id starts with a 0.
    slurmd_nodename = check_env_var(var="SLURMD_NODENAME", fun=get_node_id)
    len_id = len(str(get_lowest_node_id()))
    return slurmd_nodename[-len_id:]


def get_node_name() -> str:
    return check_env_var(var="SLURMD_NODENAME", fun=get_node_id)


def is_first_node() -> bool:
    return get_node_id() == get_lowest_node_id()


def is_on_slurm_cluster() -> bool:
    return "SLURM_JOB_ID" in os.environ


def check_env_var(var: str, fun: Optional[Callable[..., Any]] = None) -> str:
    value = os.getenv(var)
    if value is None:
        suffix = ""
        if fun is not None:
            suffix = f" before calling `{fun.__name__}`"
        error_msg = f"Environment variable '{var}' must be set" + suffix + "."
        raise KaraboDaskError(error_msg)
    return value
