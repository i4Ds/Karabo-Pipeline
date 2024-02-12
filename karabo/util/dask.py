"""Module for dask-related functionality."""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable
from typing import Any, Callable, List, Literal, Optional, Tuple, Type, Union, cast
from warnings import warn

import psutil
from dask import compute, delayed  # type: ignore[attr-defined]
from dask.distributed import Client, LocalCluster, Nanny, Worker
from dask_mpi import initialize
from mpi4py import MPI
from typing_extensions import assert_never

from karabo.util.data_util import extract_chars_from_string, extract_digit_from_string
from karabo.util.file_handler import FileHandler
from karabo.warning import KaraboWarning


class DaskHandlerBasic:
    """Base-class for dask-handler functionality.

    Attributes
    ----------
    dask_client:
        The Dask client object. If None, a new client will be created.
    memory_limit:
        The memory_limit per worker in GB. If None, the memory limit will
        be set to the maximum available memory on the node (see documentation)
        in dask for `memory_limit`.
    n_threads_per_worker:
        The number of threads to use per worker. Standard is None, which
        means that the number of threads will be equal to the number of
        cores.
    use_dask:
        Whether to use Dask or not. If None, then Karabo will decide whether
        to use dask or not for certain tasks.
    use_processes:
        Use processes instead of threads?
        Threads:
            - Fast to initiate.
            - No need to transfer data to them.
            - Limited by the GIL, which allows one thread to read the code at once.
        Processes:
            - Take time to set up.
            - Slow to transfer data to.
            - Each have their own GIL and so don't need to take turns reading the code.
    """

    dask_client: Optional[Client] = None
    memory_limit: Optional[float] = None
    n_threads_per_worker: Optional[int] = None
    use_dask: Optional[bool] = None
    use_proccesses: bool = False  # Some packages, such as pybdsf, do not work
    # with processes because they spawn subprocesses.

    _setup_called: bool = False

    @classmethod
    def setup(cls) -> None:
        """Calls `get_dask_client`."""
        _ = cls.get_dask_client()
        cls._setup_called = True

    @classmethod
    def get_dask_client(cls) -> Client:
        """Get (create if not exists) a dask-client.

        Returns:
            Dask-client.
        """
        if cls.dask_client is not None:
            return cls.dask_client
        if MPI.COMM_WORLD.Get_size() > 1:  # TODO: testing of whole if-block
            n_threads_per_worker = cls.n_threads_per_worker
            if n_threads_per_worker is None:
                initialize(comm=MPI.COMM_WORLD)
            else:
                initialize(nthreads=n_threads_per_worker, comm=MPI.COMM_WORLD)
            cls.dask_client = Client(processes=cls.use_proccesses)
            if MPI.COMM_WORLD.rank == 0:
                print(f"Dashboard link: {cls.dask_client.dashboard_link}", flush=True)
                atexit.register(cls._dask_cleanup)
        else:
            cls.dask_client = cls._get_local_dask_client()
            # Register cleanup function
            print(f"Dashboard link: {cls.dask_client.dashboard_link}", flush=True)
            atexit.register(cls._dask_cleanup)
        return cls.dask_client

    @classmethod
    def should_dask_be_used(cls, override: Optional[bool] = None) -> bool:
        """Util function to decide whether dask should be used or not.

        Args:
            override: Override? Has highest priority.

        Returns:
            Decision whether dask should be used or not.
        """
        if override is not None:
            return override
        elif cls.use_dask is not None:
            return cls.use_dask
        elif cls.dask_client is not None:
            return True
        else:
            return False

    @classmethod
    def parallelize_with_dask(
        cls,
        iterate_function: Callable[..., Any],
        iterable: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Any, Tuple[Any, ...], List[Any]]:
        """
        Run a function over an iterable in parallel using dask, and gather the results.

        args & kwargs will get passed to `Delayed`.

        Args:
            iterate_function: The function to be applied to each element of `iterable`.
                The function takes the current element of the iterable as its first
                argument, followed by any positional arguments, and then any keyword
                arguments.

            iterable
                The iterable over which the function will be applied. Each element of
                `iterable` will be passed to `iterate_function`.

        Returns: A tuple containing the results of the `iterate_function` for each
            element in the iterable. The results are gathered using dask's `compute`
            function.
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
    def _dask_cleanup(cls) -> None:
        """Shutdown & close `cls.dask_client`."""
        if cls.dask_client is not None:
            cls.dask_client.shutdown()
            cls.dask_client.close()

    @classmethod
    def _calc_num_of_workers(cls) -> int:
        """Estimates the number of workers considering settings and availability.

        Returns:
            Etimated number of workers.
        """
        memory_limit = cls.memory_limit
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
    def _get_local_dask_client(cls) -> Client:
        """Creates a local dask-client.

        Returns:
            Created dask-client.
        """
        n_workers = cls._calc_num_of_workers()
        client = Client(
            n_workers=n_workers,
            threads_per_worker=cls.n_threads_per_worker,
            processes=cls.use_proccesses,
        )
        return client


class DaskHandlerSlurm(DaskHandlerBasic):
    """Dask-handler for slurm-based jobs.

    Attributes
    ----------
    use_workers_or_nannies:
        Whether to use workers or nannies (default).
        This could lead to more processing (see documentation for dask usage
        in Karabo).
    n_workers_scheduler_node : int
        The number of workers to start on the scheduler node.
    timeout: int
        Timeout in seconds for the dask-scheduler to wait for all the
        workers to connect.
    """

    use_workers_or_nannies: Literal["workers", "nannies"] = "nannies"
    n_workers_scheduler_node: int = 1
    # with processes because they spawn subprocesses.
    timeout: int = 60

    _nodes_prepared: bool = False

    @classmethod
    def get_dask_client(cls) -> Client:
        """Get (create if not exists) a dask-client for a SLURM environment.

        Returns:
            Dask-client.
        """
        dask_client = cls.dask_client
        if dask_client is not None:
            return dask_client
        if not cls._setup_called and cls.is_first_node():
            cls.setup()
        if cls.get_number_of_nodes() > 1:
            dask_client = cast(  # dask_client is None if not first-node
                Client,  # however, needed workaround to keep api-compatibility
                cls._setup_dask_for_slurm(),
            )
            if dask_client is not None:
                cls.dask_client = dask_client
            return dask_client
        else:
            cls.dask_client = super(DaskHandlerSlurm, cls).get_dask_client()
        return cls.dask_client

    @classmethod
    def should_dask_be_used(cls, override: Optional[bool] = None) -> bool:
        """Util function to decide whether dask should be used or not.

        This implementation differs a bit from the basic-class, where
            on SLURM-systems, additional checks are taken into consideration.

        Args:
            override: Override? Has highest priority.

        Returns:
            Decision whether dask should be used or not.
        """
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
    def _dask_cleanup(cls) -> None:
        """Shutdown & close `cls.dask_client`.

        In addition, `dask_info_dir` will get removed if exists.
        """
        dask_info_dir, _, _ = cls._get_dask_paths_for_slurm()
        if os.path.exists(dask_info_dir) and os.path.isdir(dask_info_dir):
            shutil.rmtree(dask_info_dir)

        super(DaskHandlerSlurm, cls)._dask_cleanup()

    @classmethod
    def _dask_info_address(cls) -> str:
        """dask_info.json path."""
        _, info_address, _ = cls._get_dask_paths_for_slurm()
        return info_address

    @classmethod
    def _dask_run_status(cls) -> str:
        """dask_run_status.txt path."""
        _, _, run_status = cls._get_dask_paths_for_slurm()
        return run_status

    @classmethod
    def _prepare_slurm_nodes_for_dask(cls) -> None:
        """Prepares slurm-nodes for dask-usage."""
        if not cls.is_on_slurm_cluster() or cls.get_number_of_nodes() <= 1:
            cls.use_dask = False
        elif (
            cls.is_first_node() and cls.dask_client is None and not cls._nodes_prepared
        ):
            cls._nodes_prepared = True
            slurm_job_nodelist = cls._get_job_nodelist()
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
    def _setup_nannies_workers_for_slurm(cls) -> None:
        """Setup nannies & workers."""
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
            elif cls.use_workers_or_nannies == "nannies":
                nanny = asyncio.run(start_nanny(scheduler_address))
                workers_or_nannies.append(nanny)
            else:
                assert_never(cls.use_workers_or_nannies)

        # Keep the process alive
        while os.path.exists(dask_run_status):
            time.sleep(1)

        # Shutdown process
        for worker_or_nanny in workers_or_nannies:
            result = asyncio.run(worker_or_nanny.close())
            if result == "OK":
                pass
            else:
                if isinstance(worker_or_nanny, Worker):
                    instance = "worker"
                else:
                    instance = "nanny"
                print(
                    f"There was an issue closing {instance} {worker_or_nanny.address}",
                    file=sys.stderr,
                )

        # Stop the script successfully
        sys.exit(0)

    @classmethod
    def _setup_dask_for_slurm(cls) -> Optional[Client]:
        """Setup dask for slurm.

        Returns:
            A dask-client if it's the first node, otherwise None.
        """
        if cls.is_first_node():
            _, dask_info_address, dask_run_status = cls._get_dask_paths_for_slurm()
            # Create file to show that the run is still ongoing
            with open(dask_run_status, "w") as f:
                f.write("ongoing")

            # Create client and scheduler
            cluster = LocalCluster(
                ip=cls.get_node_name(),
                n_workers=cls.n_workers_scheduler_node,
                threads_per_worker=cls.n_threads_per_worker,
            )
            dask_client = Client(cluster, proccesses=cls.use_proccesses)

            # Calculate number of workers per node
            n_workers_per_node = cls._calc_num_of_workers()

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
            ) * n_workers_per_node + cls.n_workers_scheduler_node

            dask_client.wait_for_workers(
                n_workers=n_workers_requested, timeout=cls.timeout
            )

            print(
                f"All {len(dask_client.scheduler_info()['workers'])} workers connected!"
            )
            return dask_client

        else:
            cls._setup_nannies_workers_for_slurm()
            return None

    @classmethod
    def _get_dask_paths_for_slurm(cls) -> Tuple[str, str, str]:
        """Gets dask-file paths for slurm setup.

        This needs to be a function, to enable the `FileHandler` lazy path-loading,
        hence allowing path-changes at run-time.

        Returns:
            dask_info_dir, dask-info-address, dask-run-status
        """
        slurm_job_id = cls._get_job_id()
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
    def _extract_node_ids_from_node_list(cls) -> List[int]:
        """Extracts all node-ids of the current slurm-job as a list.

        Returns:
            Node-ids.
        """
        slurm_job_nodelist = cls._get_job_nodelist()
        if cls.get_number_of_nodes() == 1:
            # Node name will be something like "psanagpu115"
            return [extract_digit_from_string(slurm_job_nodelist)]
        node_list = slurm_job_nodelist.split("[")[1].split("]")[0]
        id_ranges = node_list.split(",")
        node_ids: List[int] = []
        for id_range in id_ranges:
            if "-" in id_range:
                min_id, max_id = id_range.split("-")
                node_ids += [i for i in range(int(min_id), int(max_id) + 1)]
            else:
                node_ids.append(int(id_range))

        return node_ids

    @classmethod
    def _get_min_max_of_node_id(cls) -> Tuple[int, int]:
        """Returns the min max from SLURM_JOB_NODELIST.

        Returns:
            Min & Max node-ids.
        """
        node_list = cls._extract_node_ids_from_node_list()
        return min(node_list), max(node_list)

    @classmethod
    def _get_lowest_node_id(cls) -> int:
        """Get the lowest slurm node-id.

        Returns:
            Lowest node-id.
        """
        return cls._get_min_max_of_node_id()[0]

    @classmethod
    def _get_base_string_node_list(cls) -> str:
        """Gets the node-list base-string.

        Returns:
            Node-list base-string.
        """
        slurm_job_nodelist = cls._get_job_nodelist()
        if cls.get_number_of_nodes() == 1:
            return extract_chars_from_string(slurm_job_nodelist)
        else:
            return slurm_job_nodelist.split("[")[0]

    @classmethod
    def _get_lowest_node_name(cls) -> str:
        """Gets the lowest node-name.

        Returns:
            Lowest node-name.
        """
        return cls._get_base_string_node_list() + str(cls._get_lowest_node_id())

    @classmethod
    def get_number_of_nodes(cls) -> int:
        """Gets the number of nodes of the slurm-job.

        Returns:
            Number of nodes.
        """
        n_nodes = os.environ["SLURM_JOB_NUM_NODES"]
        return int(n_nodes)

    @classmethod
    def get_node_id(cls) -> int:
        """Gets the current node-id.

        Returns:
            Node-id.
        """
        # Attention, often the node id starts with a 0.
        slurmd_nodename = cls.get_node_name()
        len_id = len(cls._get_base_string_node_list())
        return int(slurmd_nodename[-len_id:])

    @classmethod
    def get_node_name(cls) -> str:
        """Gets the current node-name.

        Returns:
            Node-name.
        """
        return os.environ["SLURMD_NODENAME"]

    @classmethod
    def is_first_node(cls) -> bool:
        """Util function to check if current-node is fist-node.

        Returns:
            Check-result.
        """
        return cls.get_node_id() == cls._get_lowest_node_id()

    @classmethod
    def _get_job_nodelist(cls) -> str:
        """Gets the nodelist of the current job as an `str`.

        Returns:
            Nodelist of current job.
        """
        return os.environ["SLURM_JOB_NODELIST"]

    @classmethod
    def _get_job_id(cls) -> str:
        """Gets the current job-id as an `str`.

        Returns:
            Job-id.
        """
        return os.environ["SLURM_JOB_ID"]

    @classmethod
    def is_on_slurm_cluster(cls) -> bool:
        """Util function to check if code is running in a slurm-job.

        Returns:
            Check-result.
        """
        return "SLURM_JOB_ID" in os.environ


def _select_dask_handler() -> Type[DaskHandlerBasic]:
    """Selects a dask-handler class.

    Returns:
        Chosen dask-handler class.
    """
    if DaskHandlerSlurm.is_on_slurm_cluster():
        return DaskHandlerSlurm
    return DaskHandlerBasic


class DaskHandler(DaskHandlerBasic):
    """Public & dev API for dask associated functionality.

    This is the public dask-api for Karabo, where you don't have to worry which
    dask-handler of this module to use. You can do almost everything through this
    class. The only exception is, if you want to adjust the default settings on
    a slurm-system (customization through `DaskHandlerSlurm`).

    Attributes
    ----------
    dask_client:
        The Dask client object. If None, a new client will be created.
    memory_limit:
        The memory_limit per worker in GB. If None, the memory limit will
        be set to the maximum available memory on the node (see documentation)
        in dask for `memory_limit`.
    n_threads_per_worker:
        The number of threads to use per worker. Standard is None, which
        means that the number of threads will be equal to the number of
        cores.
    use_dask:
        Whether to use Dask or not. If None, then Karabo will decide whether
        to use dask or not for certain tasks.
    use_processes:
        Use processes instead of threads?
        Threads:
            - Fast to initiate.
            - No need to transfer data to them.
            - Limited by the GIL, which allows one thread to read the code at once.
        Processes:
            - Take time to set up.
            - Slow to transfer data to.
            - Each have their own GIL and so don't need to take turns reading the code.
    """

    # Important: API-functions of `DaskHandler` should redirect ALL functions defined
    # in `DaskHandlerBasic` through `_handler`. This ensures that in case `_handler`
    # is a more specific implementation of `DaskHandlerBasic`, that the according
    # overwritten functions will be used instead.
    _handler = _select_dask_handler()

    @classmethod
    def setup(cls) -> None:
        """Calls `get_dask_client`."""
        return cls._handler.setup()

    @classmethod
    def get_dask_client(cls) -> Client:
        """Get (create if not exists) a dask-client.

        Returns:
            Dask-client.
        """
        return cls._handler.get_dask_client()

    @classmethod
    def should_dask_be_used(cls, override: Optional[bool] = None) -> bool:
        """Util function to decide whether dask should be used or not.

        Args:
            override: Override? Has highest priority.

        Returns:
            Decision whether dask should be used or not.
        """
        return cls._handler.should_dask_be_used(override)

    @classmethod
    def parallelize_with_dask(
        cls,
        iterate_function: Callable[..., Any],
        iterable: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Any, Tuple[Any, ...], List[Any]]:
        """
        Run a function over an iterable in parallel using dask, and gather the results.

        args & kwargs will get passed to `Delayed`.

        Args:
            iterate_function: The function to be applied to each element of `iterable`.
                The function takes the current element of the iterable as its first
                argument, followed by any positional arguments, and then any keyword
                arguments.

            iterable
                The iterable over which the function will be applied. Each element of
                `iterable` will be passed to `iterate_function`.

        Returns: A tuple containing the results of the `iterate_function` for each
            element in the iterable. The results are gathered using dask's `compute`
            function.
        """
        return cls._handler.parallelize_with_dask(
            iterate_function,
            iterable,
            *args,
            **kwargs,
        )

    @classmethod
    def _dask_cleanup(cls) -> None:
        """Shutdown & close `cls.dask_client`."""
        return cls._handler._dask_cleanup()

    @classmethod
    def _calc_num_of_workers(cls) -> int:
        """Estimates the number of workers considering settings and availability.

        Returns:
            Etimated number of workers.
        """
        return cls._handler._calc_num_of_workers()

    @classmethod
    def _get_local_dask_client(cls) -> Client:
        """Creates a local dask-client.

        Returns:
            Created dask-client.
        """
        return cls._handler._get_local_dask_client()
