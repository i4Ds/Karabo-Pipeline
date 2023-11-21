from __future__ import annotations

import asyncio
import atexit
import json
import os
import sys
import time
from collections.abc import Iterable
from typing import Any, Callable, List, Optional, Tuple, Union

import psutil
from dask import compute, delayed  # type: ignore[attr-defined]
from dask.distributed import Client, LocalCluster, Nanny, Worker

from karabo.error import KaraboDaskError
from karabo.util._types import IntFloat
from karabo.util.data_util import extract_chars_from_string, extract_digit_from_string
from karabo.warning import KaraboWarning

DASK_INFO_FOLDER = ".karabo_dask"
DASK_INFO_FILE = "dask_info.json"
DASK_RUN_STATUS = "dask_run_status.txt"

##
if "SLURM_JOB_ID" in os.environ:
    DASK_INFO_FOLDER = os.path.join(DASK_INFO_FOLDER, str(os.environ["SLURM_JOB_ID"]))
os.makedirs(DASK_INFO_FOLDER, exist_ok=True)
DASK_INFO_ADDRESS = os.path.join(DASK_INFO_FOLDER, DASK_INFO_FILE)
DASK_RUN_STATUS = os.path.join(DASK_INFO_FOLDER, DASK_RUN_STATUS)


class DaskHandler:
    """
    A class for managing a Dask client.

    Attributes
    ----------
    dask_client: Optional[Client]
        The Dask client object. If None, a new client will be created.
    n_workers_scheduler_node : int
        The number of workers to start on the scheduler node.
    memory_limit : Optional[float]
        The memory_limit per worker in GB. If None, the memory limit will
        be set to the maximum available memory on the node (see documentation)
        in dask for `memory_limit`.
    n_threads_per_worker : int
        The number of threads to use per worker. Standard is None, which
        means that the number of threads will be equal to the number of
        cores.
    use_dask: Optional[bool]
        Whether to use Dask or not. If None, Dask will be used if the
        current node is a SLURM node and there are more than 1 node.
    use_workers_or_nannies: Optional[str]
        Whether to use workers or nannies. If None, nannies will be used.
        This could lead to more processing (see documentation for dask usage
        in Karabo).
    TIMEOUT: int
        The timeout in seconds for the Dask scheduler to wait for all the
        workers to connect.


    Methods
    -------
    setup() -> None:
        Sets up the Dask client. If the client does not exist, and the
        current node is a SLURM node and there are more than 1 node, a
        Dask client will be created but not returned. Then, when a function
        can make use of dask, it will make use of dask automatically. This
        function need to be only called once at the beginning of the script.
        It stops the processing of the script if the script is not running on the
        main node.
    get_dask_client() -> Client:
        Returns a Dask client object. If the client does not exist, and
        the current node is a SLURM node and there are more than 1 node,
        a Dask client will be created.



    """

    dask_client: Optional[Client] = None
    n_workers_scheduler_node: int = 1
    memory_limit: Optional[int] = None
    n_threads_per_worker: Optional[int] = None
    use_dask: Optional[bool] = None
    use_workers_or_nannies: Optional[str] = "nannies"
    TIMEOUT: int = 60

    # Some internal variables
    _nodes_prepared: bool = False
    _setup_called: bool = False

    @staticmethod
    def setup() -> None:
        _ = DaskHandler.get_dask_client()
        DaskHandler._setup_called = True

    @staticmethod
    def get_dask_client() -> Client:
        # Get IS_DOCKER_CONTAINER variable
        if os.environ.get("IS_DOCKER_CONTAINER", "false").lower() == "true":
            from dask.distributed import Client
            from dask_mpi import initialize
            from mpi4py import MPI

            # mpi4py.MPI.Intracomm
            n_threads_per_worker = DaskHandler.n_threads_per_worker
            if n_threads_per_worker is None:  # ugly hotfix to be able to initialize
                initialize(comm=MPI.COMM_WORLD)
            else:
                initialize(nthreads=n_threads_per_worker, comm=MPI.COMM_WORLD)
            initialize(nthreads=n_threads_per_worker, comm=MPI.COMM_WORLD)
            DaskHandler.dask_client = Client()
        elif DaskHandler.dask_client is None:
            if (
                not DaskHandler._setup_called
                and is_on_slurm_cluster()
                and is_first_node()
            ):
                print(
                    KaraboWarning(
                        "DaskHandler.setup() has to be called at the beginning "
                        "of the script. This could lead to unexpected behaviour "
                        "on a SLURM cluster if not (see documentation)."
                    )
                )
            if is_on_slurm_cluster() and get_number_of_nodes() > 1:
                DaskHandler.dask_client = setup_dask_for_slurm(
                    DaskHandler.n_workers_scheduler_node,
                    DaskHandler.memory_limit,
                )
            else:
                DaskHandler.dask_client = get_local_dask_client(
                    DaskHandler.memory_limit
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
        elif is_on_slurm_cluster() and get_number_of_nodes() > 1:
            return True
        else:
            return False


def parallelize_with_dask(
    iterate_function: Callable[..., Any],
    iterable: Iterable[Any],
    *args: Any,
    **kwargs: Any,
) -> Union[Any, Tuple[Any, ...], List[Any]]:
    """
    Run a function over an iterable in parallel using Dask, and gather the results.

    Parameters
    ----------
    iterate_function : callable
        The function to be applied to each element of the iterable. The function should
        take the current element of the iterable as its first argument, followed by any
        positional arguments, and then any keyword arguments.

    iterable : iterable
        The iterable over which the function will be applied. Each element of this
        iterable will be passed to the `iterate_function`.

    *args : tuple
        Positional arguments that will be passed to the `iterate_function` after the
        current element of the iterable.

    **kwargs : dict
        Keyword arguments that will be passed to the `iterate_function`.

    Returns
    -------
    tuple
        A tuple containing the results of the `iterate_function` for each element in the
        iterable. The results are gathered using Dask's compute function.

    Notes
    -----
    - If 'verbose' is present in **kwargs and is set to True, additional progress
    messages will be printed.
    - This function utilizes the distributed scheduler of Dask.
    """
    if not DaskHandler._setup_called:
        DaskHandler.setup()

    delayed_results = []

    for element in iterable:
        if "verbose" in kwargs and kwargs["verbose"]:
            print(f"Processing element {element}...\nExtracting data...")

        delayed_result = delayed(iterate_function)(element, *args, **kwargs)
        delayed_results.append(delayed_result)

    return compute(*delayed_results, scheduler="distributed")


def dask_cleanup(client: Client) -> None:
    # Renove run status file
    if os.path.exists(DASK_RUN_STATUS):
        os.remove(DASK_RUN_STATUS)

    # Wait for nannys to shut down
    time.sleep(10)

    # Remove the scheduler file if somehow it was not removed
    if os.path.exists(DASK_INFO_ADDRESS):
        os.remove(DASK_INFO_ADDRESS)

    # Remove the dashboard file if somehow it was not removed
    if os.path.exists("karabo-dask-dashboard.txt"):
        os.remove("karabo-dask-dashboard.txt")

    if client is not None:
        client.shutdown()
        client.close()


def prepare_slurm_nodes_for_dask() -> None:
    # Detect if we are on a slurm cluster
    if not is_on_slurm_cluster() or get_number_of_nodes() <= 1:
        DaskHandler.use_dask = False
        return
    elif (
        is_first_node()
        and DaskHandler.dask_client is None
        and not DaskHandler._nodes_prepared
    ):
        DaskHandler._nodes_prepared = True
        slurm_job_nodelist = check_env_var(
            var="SLURM_JOB_NODELIST", fun=prepare_slurm_nodes_for_dask
        )
        print(
            f"""
            Preparing SLURM nodes for dask...
            First Node, containing the scheduler, is: {get_node_name()}
            With the help of dask, the following nodes will be used:
            {slurm_job_nodelist}
            """
        )

    elif not is_first_node() and not DaskHandler._nodes_prepared:
        # TODO: Here setup_nannies_workers_for_slurm() could be called
        # but there is no if name == main guard in this file.
        pass


def calculate_number_of_workers_per_node(
    memory_limit: Optional[IntFloat],
) -> int:
    if memory_limit is None:
        return 1
    # Calculate number of workers per node
    ram = psutil.virtual_memory().available / 1e9  # GB
    n_workers_per_node = int(ram / (memory_limit))
    if ram < memory_limit:
        KaraboWarning(
            f"Only {ram} GB of RAM available. Requested at least "
            f"{memory_limit} GB. Setting number of "
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
    memory_limit: Optional[IntFloat],
) -> Client:
    # Calculate number of workers per node
    n_workers = calculate_number_of_workers_per_node(memory_limit)
    client = Client(
        LocalCluster(
            n_workers=n_workers,
            threads_per_worker=DaskHandler.n_threads_per_worker,
        )
    )
    return client


def setup_nannies_workers_for_slurm() -> None:
    # Wait until dask info file is created
    while not os.path.exists(DASK_INFO_ADDRESS):
        time.sleep(1)

    # Load dask info file
    with open(DASK_INFO_ADDRESS, "r") as f:
        dask_info = json.load(f)

    # Calculate memory usage of each worker
    if DaskHandler.memory_limit is None:
        memory_limit = f"{psutil.virtual_memory().available / 1e9}GB"
    else:
        memory_limit = f"{DaskHandler.memory_limit}GB"

    async def start_worker(scheduler_address: str) -> Worker:
        worker = await Worker(
            scheduler_address,
            nthreads=DaskHandler.n_threads_per_worker,
            memory_limit=memory_limit,
        )
        await worker.finished()
        return worker  # type: ignore

    async def start_nanny(scheduler_address: str) -> Nanny:
        nanny = await Nanny(
            scheduler_address,
            nthreads=DaskHandler.n_threads_per_worker,
            memory_limit=memory_limit,
        )
        await nanny.finished()
        return nanny  # type: ignore

    scheduler_address = str(dask_info["scheduler_address"])
    n_workers = int(str(dask_info["n_workers_per_node"]))

    # Start workers or nannies
    workers_or_nannies: List[Union[Worker, Nanny]] = []
    for _ in range(n_workers):
        if DaskHandler.use_workers_or_nannies == "workers":
            worker = asyncio.run(start_worker(scheduler_address))
            workers_or_nannies.append(worker)
        else:
            nanny = asyncio.run(start_nanny(scheduler_address))
            workers_or_nannies.append(nanny)

    # Keep the process alive
    while os.path.exists(DASK_RUN_STATUS):
        time.sleep(1)

    # Shutdown process
    for worker_or_nanny in workers_or_nannies:
        result = asyncio.run(worker_or_nanny.close())
        if result == "OK":
            pass
        else:
            print(
                f"""
                There was an issue closing the worker or nanny at
                 {worker_or_nanny.address}
                """
            )

    # Stop the script successfully
    sys.exit(0)


def setup_dask_for_slurm(
    n_workers_scheduler_node: int,
    memory_limit: Optional[IntFloat],
) -> Client:
    if is_first_node():
        # Create file to show that the run is still ongoing
        with open(DASK_RUN_STATUS, "w") as f:
            f.write("ongoing")

        # Create client and scheduler
        cluster = LocalCluster(
            ip=get_node_name(),
            n_workers=n_workers_scheduler_node,
            threads_per_worker=DaskHandler.n_threads_per_worker,
        )
        dask_client = Client(cluster)

        # Calculate number of workers per node
        n_workers_per_node = calculate_number_of_workers_per_node(memory_limit)

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

        dask_client.wait_for_workers(
            n_workers=n_workers_requested, timeout=DaskHandler.TIMEOUT
        )

        print(f"All {len(dask_client.scheduler_info()['workers'])} workers connected!")
        return dask_client

    else:
        setup_nannies_workers_for_slurm()
        return None  # type: ignore


def extract_node_ids_from_node_list() -> List[int]:
    slurm_job_nodelist = check_env_var(
        var="SLURM_JOB_NODELIST", fun=extract_node_ids_from_node_list
    )
    if get_number_of_nodes() == 1:
        # Node name will be something like "psanagpu115"
        return [extract_digit_from_string(slurm_job_nodelist)]
    node_list = slurm_job_nodelist.split("[")[1].split("]")[0]
    id_ranges = node_list.split(",")
    node_ids = []
    for id_range in id_ranges:
        if "-" in id_range:
            min_id, max_id = id_range.split("-")
            node_ids += [i for i in range(int(min_id), int(max_id) + 1)]
        else:
            node_ids.append(int(id_range))

    return node_ids


def get_min_max_of_node_id() -> Tuple[int, int]:
    """
    Returns the min max from SLURM_JOB_NODELIST.
    Works if it's run only on two nodes (separated with a comma)
    of if it runs on more than two nodes (separated with a dash).
    """
    node_list = extract_node_ids_from_node_list()
    return min(node_list), max(node_list)


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


def get_node_id() -> int:
    # Attention, often the node id starts with a 0.
    slurmd_nodename = check_env_var(var="SLURMD_NODENAME", fun=get_node_id)
    len_id = len(get_base_string_node_list())
    return int(slurmd_nodename[-len_id:])


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
