from distributed import Client, LocalCluster
import psutil

client = None


def get_local_dask_client(min_ram_gb_per_worker: int = 4) -> Client:
    global client
    if client is not None:
        return client
    min_ram_gb_per_worker *= 1024
    cpus = psutil.cpu_count()
    ram = psutil.virtual_memory().total / 1024 / 1024
    if ram / cpus >= min_ram_gb_per_worker:
        client = Client(LocalCluster(n_workers=cpus))
    else:
        workers = cpus
        while ram / workers < min_ram_gb_per_worker:
            workers -= 1

        client = Client(LocalCluster(n_workers=workers))
    return client

