from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Callable, List, Optional, Tuple, Union

import psutil
from dask import compute, delayed  # type: ignore[attr-defined]
from dask.distributed import Client
from dask_mpi import initialize
from mpi4py import MPI

from karabo.error import KaraboDaskError
from karabo.util._types import IntFloat
from karabo.warning import KaraboWarning


class DaskHandler:
    """
    A class for managing a Dask client. This class is a singleton, meaning that
    only one instance of this class can exist at any given time. This also
    allows you to create your own client and pass it to this class.

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
    n_threads_per_worker: Optional[int] = 1
    use_dask: Optional[bool] = None
    use_workers_or_nannies: Optional[str] = "nannies"
    use_proccesses: bool = False  # Some packages, such as pybdsf, do not work
    # with processes because they spawn subprocesses.
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
        print("Number of nodes: ", get_number_of_nodes())
        if DaskHandler.should_dask_be_used():
            # Check which type of client to use. Dask MPI or a local dask client?
            if MPI.COMM_WORLD.Get_size() > 1 or get_number_of_nodes() > 1:
                n_threads_per_worker = DaskHandler.n_threads_per_worker
                initialize(nthreads=n_threads_per_worker, comm=MPI.COMM_WORLD)
                DaskHandler.dask_client = Client()
                # Write the dashboard link to a file
                with open("karabo-dask-dashboard.txt", "w") as f:
                    f.write(DaskHandler.dask_client.dashboard_link)
                # Register cleanup function
                print(f"Dashboard link: {DaskHandler.dask_client.dashboard_link}")
            else:
                client = get_local_dask_client(DaskHandler.memory_limit)
                DaskHandler.dask_client = client
        return DaskHandler.dask_client

    @staticmethod
    def should_dask_be_used(override: Optional[bool] = None) -> bool:
        if override is not None:
            return override
        elif DaskHandler.use_dask is not None:
            return DaskHandler.use_dask
        elif DaskHandler.dask_client is not None:
            return True
        elif (
            is_on_slurm_cluster() and get_number_of_nodes() > 1
        ) or MPI.COMM_WORLD.Get_size() > 1:
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
        n_workers=n_workers,
        threads_per_worker=DaskHandler.n_threads_per_worker,
    )
    return client


def get_number_of_nodes() -> int:
    n_nodes = check_env_var(var="SLURM_JOB_NUM_NODES", fun=get_number_of_nodes)
    return int(n_nodes)


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
