import os
import sys
import time
from subprocess import call

import psutil
from dask.distributed import Client, LocalCluster

SCHEDULER_ADDRESS = "scheduler_address.json"
STOP_WORKER_FILE = "stop_workers"


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


def dask_cleanup(client: Client):
    # Create the stop_workers file
    with open(STOP_WORKER_FILE, "w") as _:
        pass

    # Give some time for the workers to exit before closing the client
    time.sleep(10)

    # Remove the stop_workers file
    if os.path.exists(STOP_WORKER_FILE):
        os.remove(STOP_WORKER_FILE)

    # Remove the scheduler file
    if os.path.exists(SCHEDULER_ADDRESS):
        os.remove(SCHEDULER_ADDRESS)
    if client is not None:
        client.close()


def setup_dask_for_slurm(n_workers_scheduler_node: int = 1):
    # Detect if we are on a slurm cluster
    if not is_on_slurm_cluster() or os.getenv("SLURM_JOB_NUM_NODES") == "1":
        print("Not on a SLURM cluster or only 1 node. Not setting up dask.")
        return None

    else:
        if is_first_node():
            # Remove old scheduler file
            if os.path.exists(SCHEDULER_ADDRESS):
                os.remove(SCHEDULER_ADDRESS)

            # Create client and scheduler
            cluster = LocalCluster(
                ip=get_lowest_node_name(), n_workers=n_workers_scheduler_node
            )
            client = Client(cluster)

            # Write scheduler file
            with open(SCHEDULER_ADDRESS, "w") as f:
                f.write(cluster.scheduler_address)

            # Wait until all workers are connected
            n_workers_requested = get_number_of_nodes() - 1 + n_workers_scheduler_node
            while len(client.scheduler_info()["workers"]) < n_workers_requested:
                print(
                    f"Waiting for all workers to connect. Currently "
                    f"{len(client.scheduler_info()['workers'])} "
                    f"workers connected of {n_workers_requested} requested."
                )
                time.sleep(1)

            print(f"All {len(client.scheduler_info()['workers'])} workers connected!")
            return client

        else:
            # Wait some time to make sure the scheduler file is new
            time.sleep(5)

            # Wait until scheduler file is created
            while not os.path.exists(SCHEDULER_ADDRESS):
                print("Waiting for scheduler file to be created.")
                time.sleep(1)

            # Read scheduler file
            with open(SCHEDULER_ADDRESS, "r") as f:
                scheduler_address = f.read()

            # Create client
            call(["dask", "worker", scheduler_address])

            # Run until stop_workers file is created
            while True:
                if os.path.exists(STOP_WORKER_FILE):
                    print("Stop workers file detected. Exiting.")
                    sys.exit(0)
                time.sleep(5)


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


def get_current_time():
    return time.strftime("%H:%M:%S", time.localtime())


def is_on_slurm_cluster():
    return "SLURM_JOB_ID" in os.environ
