from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable
from typing import Any, Callable, List, Optional, Tuple, Type, Union, cast
from warnings import warn

import psutil
from dask import compute, delayed  # type: ignore[attr-defined]
from dask.distributed import Client, LocalCluster, Nanny, Worker
from dask_mpi import initialize
from mpi4py import MPI

from karabo.util._types import IntFloat
from karabo.util.data_util import extract_chars_from_string, extract_digit_from_string
from karabo.util.file_handler import FileHandler
from karabo.warning import KaraboWarning


def fetch_dask_handler() -> Union[Type[DaskHandler], Type[DaskSlurmHandler]]:
    """Utility function to automatically choose a Handler.

    Returns:
        The chosen Handler.
    """
    if DaskSlurmHandler.is_on_slurm_cluster():
        return DaskSlurmHandler
    return DaskHandler


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
    """

    dask_client: Optional[Client] = None
    n_workers_scheduler_node: int = 1
    memory_limit: Optional[int] = None
    n_threads_per_worker: Optional[int] = None
    use_dask: Optional[bool] = None
    use_workers_or_nannies: Optional[str] = "nannies"
    use_proccesses: bool = False  # Some packages, such as pybdsf, do not work
    # with processes because they spawn subprocesses.
    TIMEOUT: int = 60

    # Some internal variables
    _nodes_prepared: bool = False
    _setup_called: bool = False

    @classmethod
    def setup(cls) -> None:
        _ = cls.get_dask_client()
        cls._setup_called = True

    @classmethod
    def get_dask_client(cls) -> Client:
        if cls.dask_client is not None:
            return cls.dask_client
        if MPI.COMM_WORLD.Get_size() > 1:  # TODO: testing of whole if-block
            n_threads_per_worker = cls.n_threads_per_worker
            if n_threads_per_worker is None:
                initialize(comm=MPI.COMM_WORLD)
            else:
                initialize(nthreads=n_threads_per_worker, comm=MPI.COMM_WORLD)
            cls.dask_client = Client(processes=cls.use_proccesses)  # TODO: testing
            if MPI.COMM_WORLD.rank == 0:
                print(f"Dashboard link: {cls.dask_client.dashboard_link}", flush=True)
                atexit.register(cls.dask_cleanup, cls.dask_client)
        else:
            cls.dask_client = cls.get_local_dask_client(cls.memory_limit)
            # Register cleanup function
            print(f"Dashboard link: {cls.dask_client.dashboard_link}", flush=True)
            atexit.register(cls.dask_cleanup, cls.dask_client)
        return cls.dask_client

    @classmethod
    def should_dask_be_used(cls, override: Optional[bool] = None) -> bool:
        if override is not None:
            return override
        elif cls.use_dask is not None:
            return cls.use_dask
        elif cls.dask_client is not None:
            return True
        else:
            return False

    @classmethod
    def calc_num_of_workers(
        cls,
        memory_limit: Optional[IntFloat],
    ) -> int:
        """Estimates the number of workers considering settings and availability.

        Args:
            memory_limit: Memory constraint.

        Returns:
            Etimated number of workers.
        """
        if memory_limit is None:
            return 1
        # Calculate number of workers
        ram = psutil.virtual_memory().available / 1e9  # GB
        n_workers = int(ram / memory_limit)
        if ram < memory_limit:
            warn(
                KaraboWarning(
                    f"Only {ram} GB of RAM available. Requested at least "
                    + f"{memory_limit} GB. Setting number of "
                    + "workers to 1."
                )
            )
            n_workers = 1

        if n_workers > (cpu_count := psutil.cpu_count()):
            warn(
                KaraboWarning(
                    f"Only {cpu_count} CPUs available. Requested "
                    + f"{n_workers} workers per node. Setting number of "
                    + f"workers to {cpu_count}."
                )
            )
            n_workers = cpu_count

        return n_workers

    @classmethod
    def get_local_dask_client(
        cls,
        memory_limit: Optional[IntFloat],
    ) -> Client:
        # Calculate number of workers per node
        n_workers = cls.calc_num_of_workers(memory_limit)
        client = Client(
            n_workers=n_workers,
            threads_per_worker=cls.n_threads_per_worker,
            processes=cls.use_proccesses,
        )
        return client

    @classmethod
    def parallelize_with_dask(
        cls,
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
            The function to be applied to each element of the iterable. The function
            should take the current element of the iterable as its first argument,
            followed by any positional arguments, and then any keyword arguments.

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
            A tuple containing the results of the `iterate_function` for each element in
            the iterable. The results are gathered using Dask's compute function.

        Notes
        -----
        - If 'verbose' is present in **kwargs and is set to True, additional progress
        messages will be printed.
        - This function utilizes the distributed scheduler of Dask.
        """
        if not cls._setup_called:
            cls.setup()

        delayed_results = []

        for element in iterable:
            if "verbose" in kwargs and kwargs["verbose"]:
                print(f"Processing element {element}...\nExtracting data...")

            delayed_result = delayed(iterate_function)(element, *args, **kwargs)
            delayed_results.append(delayed_result)

        return compute(*delayed_results, scheduler="distributed")

    @classmethod
    def dask_cleanup(cls, client: Client) -> None:
        if client is not None:
            client.shutdown()
            client.close()


class DaskSlurmHandler(DaskHandler):
    """Dask & Slurm related functionality resides here."""

    @classmethod
    def dask_info_address(cls) -> str:
        """dask_info.json path"""
        _, info_address, _ = cls._get_dask_paths_for_slurm()
        return info_address

    @classmethod
    def dask_run_status(cls) -> str:
        """dask_run_status.txt path"""
        _, _, run_status = cls._get_dask_paths_for_slurm()
        return run_status

    @classmethod
    def get_dask_client(cls) -> Client:
        dask_client = cls.dask_client
        if dask_client is not None:
            return dask_client
        if not cls._setup_called and cls.is_first_node():
            cls.setup()
        if cls.get_number_of_nodes() > 1:
            dask_client = cast(  # hacky workaround
                Client,
                cls.setup_dask_for_slurm(
                    cls.n_workers_scheduler_node,
                    cls.memory_limit,
                ),
            )
            if dask_client is not None:
                cls.dask_client = dask_client
            return dask_client
        else:
            cls.dask_client = super(DaskSlurmHandler, cls).get_dask_client()
        return cls.dask_client

    @classmethod
    def prepare_slurm_nodes_for_dask(cls) -> None:
        # Detect if we are on a slurm cluster
        if not cls.is_on_slurm_cluster() or cls.get_number_of_nodes() <= 1:
            cls.use_dask = False
            return
        elif (
            cls.is_first_node() and cls.dask_client is None and not cls._nodes_prepared
        ):
            cls._nodes_prepared = True
            slurm_job_nodelist = cls.get_job_nodelist()
            slurm_node_name = cls.get_node_name()
            print(
                f"""
                Preparing SLURM nodes for dask...
                First Node, containing the scheduler, is: {slurm_node_name}
                With the help of dask, the following nodes will be used:
                {slurm_job_nodelist}
                """
            )

        elif not cls.is_first_node() and not cls._nodes_prepared:
            # TODO: Here setup_nannies_workers_for_slurm() could be called
            # but there is no if name == main guard in this file.
            pass

    @classmethod
    def setup_nannies_workers_for_slurm(cls) -> None:
        # Wait until dask info file is created
        _, dask_info_address, dask_run_status = cls._get_dask_paths_for_slurm()
        while not os.path.exists(dask_info_address):
            time.sleep(1)

        # Load dask info file
        with open(dask_info_address, "r") as f:
            dask_info = json.load(f)

        # Calculate memory usage of each worker
        if cls.memory_limit is None:
            memory_limit = f"{psutil.virtual_memory().available / 1e9}GB"
        else:
            memory_limit = f"{cls.memory_limit}GB"

        async def start_worker(scheduler_address: str) -> Worker:
            worker = await Worker(
                scheduler_address,
                nthreads=cls.n_threads_per_worker,
                memory_limit=memory_limit,
            )
            await worker.finished()
            return worker  # type: ignore[no-any-return]

        async def start_nanny(scheduler_address: str) -> Nanny:
            nanny = await Nanny(
                scheduler_address,
                nthreads=cls.n_threads_per_worker,
                memory_limit=memory_limit,
            )
            await nanny.finished()
            return nanny  # type: ignore[no-any-return]

        scheduler_address = str(dask_info["scheduler_address"])
        n_workers = int(str(dask_info["n_workers_per_node"]))

        # Start workers or nannies
        workers_or_nannies: List[Union[Worker, Nanny]] = []
        for _ in range(n_workers):
            if cls.use_workers_or_nannies == "workers":
                worker = asyncio.run(start_worker(scheduler_address))
                workers_or_nannies.append(worker)
            else:
                nanny = asyncio.run(start_nanny(scheduler_address))
                workers_or_nannies.append(nanny)

        # Keep the process alive
        while os.path.exists(dask_run_status):
            time.sleep(1)

        # Shutdown process
        for worker_or_nanny in workers_or_nannies:
            result = asyncio.run(worker_or_nanny.close())
            if result == "OK":
                pass
            else:
                print(
                    "There was an issue closing the worker or nanny at "
                    + f"{worker_or_nanny.address}"
                )

        # Stop the script successfully
        sys.exit(0)

    @classmethod
    def setup_dask_for_slurm(
        cls,
        n_workers_scheduler_node: int,
        memory_limit: Optional[IntFloat],
    ) -> Optional[Client]:
        if cls.is_first_node():
            _, dask_info_address, dask_run_status = cls._get_dask_paths_for_slurm()
            # Create file to show that the run is still ongoing
            with open(dask_run_status, "w") as f:
                f.write("ongoing")

            # Create client and scheduler
            cluster = LocalCluster(
                ip=cls.get_node_name(),
                n_workers=n_workers_scheduler_node,
                threads_per_worker=cls.n_threads_per_worker,
            )
            dask_client = Client(cluster, proccesses=cls.use_proccesses)

            # Calculate number of workers per node
            n_workers_per_node = cls.calc_num_of_workers(memory_limit)

            # Create dictionary with the information
            dask_info = {
                "scheduler_address": cluster.scheduler_address,
                "n_workers_per_node": n_workers_per_node,
            }

            # Write scheduler file
            with open(dask_info_address, "w") as f:
                json.dump(dask_info, f)

            # Wait until all workers are connected
            n_workers_requested = (
                cls.get_number_of_nodes() - 1
            ) * n_workers_per_node + n_workers_scheduler_node

            dask_client.wait_for_workers(
                n_workers=n_workers_requested, timeout=cls.TIMEOUT
            )

            print(
                f"All {len(dask_client.scheduler_info()['workers'])} workers connected!"
            )
            return dask_client

        else:
            cls.setup_nannies_workers_for_slurm()
            return None

    @classmethod
    def _get_dask_paths_for_slurm(cls) -> Tuple[str, str, str]:
        """Gets dask-file paths for SLURM setup.

        This needs to be a function, to enable the `FileHandler` lazy path-loading,
        hence allowing path-changes at run-time.

        Returns:
            dask_info_dir, dask-info-address, dask-run-status
        """
        slurm_job_id = cls.get_job_id()
        prefix = f"-dask-info-slurm-{slurm_job_id}-"
        dask_info_dir = FileHandler().get_tmp_dir(
            prefix=prefix,
            purpose=f"dask-info-slurm-{slurm_job_id} disk-cache",
            mkdir=True,
            seed=slurm_job_id,
        )
        dask_info_address = os.path.join(dask_info_dir, "dask_info.json")
        dask_run_status = os.path.join(dask_info_dir, "dask_run_status.txt")
        return dask_info_dir, dask_info_address, dask_run_status

    @classmethod
    def extract_node_ids_from_node_list(cls) -> List[int]:
        slurm_job_nodelist = cls.get_job_nodelist()
        if cls.get_number_of_nodes() == 1:
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

    @classmethod
    def dask_cleanup(cls, client: Client) -> None:
        dask_info_dir, _, _ = cls._get_dask_paths_for_slurm()
        if os.path.exists(dask_info_dir) and os.path.isdir(dask_info_dir):
            shutil.rmtree(dask_info_dir)

        super(DaskSlurmHandler, cls).dask_cleanup(client=client)

    @classmethod
    def should_dask_be_used(cls, override: Optional[bool] = None) -> bool:
        if override is not None:
            return override
        elif cls.use_dask is not None:
            return cls.use_dask
        elif cls.dask_client is not None:
            return True
        elif cls.is_on_slurm_cluster() and cls.get_number_of_nodes() > 1:
            return True
        else:
            return False

    @classmethod
    def get_min_max_of_node_id(cls) -> Tuple[int, int]:
        """
        Returns the min max from SLURM_JOB_NODELIST.
        Works if it's run only on two nodes (separated with a comma)
        of if it runs on more than two nodes (separated with a dash).
        """
        node_list = cls.extract_node_ids_from_node_list()
        return min(node_list), max(node_list)

    @classmethod
    def get_lowest_node_id(cls) -> int:
        return cls.get_min_max_of_node_id()[0]

    @classmethod
    def get_base_string_node_list(cls) -> str:
        slurm_job_nodelist = cls.get_job_nodelist()
        if cls.get_number_of_nodes() == 1:
            return extract_chars_from_string(slurm_job_nodelist)
        else:
            return slurm_job_nodelist.split("[")[0]

    @classmethod
    def get_lowest_node_name(cls) -> str:
        return cls.get_base_string_node_list() + str(cls.get_lowest_node_id())

    @classmethod
    def get_number_of_nodes(cls) -> int:
        n_nodes = os.environ["SLURM_JOB_NUM_NODES"]
        return int(n_nodes)

    @classmethod
    def get_node_id(cls) -> int:
        # Attention, often the node id starts with a 0.
        slurmd_nodename = cls.get_node_name()
        len_id = len(cls.get_base_string_node_list())
        return int(slurmd_nodename[-len_id:])

    @classmethod
    def get_node_name(cls) -> str:
        return os.environ["SLURMD_NODENAME"]

    @classmethod
    def is_first_node(cls) -> bool:
        return cls.get_node_id() == cls.get_lowest_node_id()

    @classmethod
    def get_job_nodelist(cls) -> str:
        return os.environ["SLURM_JOB_NODELIST"]

    @classmethod
    def get_job_id(cls) -> str:
        return os.environ["SLURM_JOB_ID"]

    @classmethod
    def is_on_slurm_cluster(cls) -> bool:
        return "SLURM_JOB_ID" in os.environ
