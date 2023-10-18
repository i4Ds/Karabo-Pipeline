# Containers

## Docker Images

We provide for the Karabo-pipeline [Docker images](https://www.docker.com/resources/what-container/#:~:text=A%20Docker%20container%20image%20is,tools%2C%20system%20libraries%20and%20settings.) which are hosted by the [ghcr.io](https://github.com/features/packages) registry. An overview of all available images is [here](https://github.com/i4ds/Karabo-Pipeline/pkgs/container/karabo-pipeline), if a specific version and not simply `latest` is desired. Starting from `karabo@v0.15.0`, all versions should be available. Provided you have docker, the image can be pulled as follows:

```shell
docker pull ghcr.io/i4ds/karabo-pipeline:latest
```

Docker images have the advantage that the packages needed for Karabo-pipeline are already pre-installed and you can usually run them on other operating systems. So in case the dependency resolvement of older Karabo installations is not up to date anymore, with Docker images you don't have to worry as the installation process has already been performed. In addition, Docker images can easily transform into other containers like Singularity or Sarus, which are often used in HPC-clusters.

## Launch a Docker Container

What the possibilities using Docker are is far too extensive to describe here. We refer to the official [Docker reference](https://docs.docker.com/reference/) for this. We only show here a minimal example of how Docker could be used, so you can use e.g. a [Jupyter Notebook](https://jupyter.org/) with sample code and an existing Karabo environment.

```shell
docker run -it --rm -p 8888:8888 ghcr.io/i4ds/karabo-pipeline:latest
```

This starts the Docker container of the image interactively, where we forward port 8888. After that, we start the jupyter service in the container with the following command:

```shell
jupyter lab --ip 0.0.0.0 --no-browser --port=8888 --allow-root
```

This will start a server on the same port as forwarded. Then copy the url which is given at the bottom and replace `hostname` with `localhost` and open it in a browser.

## Singularity Container

Singularity containers are often standard on HPC clusters, which do not require special permissions (unlike Docker).
We do not provide ready-made [Singularity containers](https://sylabs.io/). However, they can be easily created from Docker images with the following command (may take a while):

```shell
singularity pull docker://ghcr.io/i4ds/karabo-pipeline:latest
```

How to use Singularity containers (e.g. mount directories or enable gpu-support) can be seen in the [Singularity documentation](https://docs.sylabs.io/guides/3.1/user-guide/cli.html).

## Sarus Container

On CSCS, it is recommended to use [Sarus containers](https://sarus.readthedocs.io/en/stable/index.html) (see CSCS [Sarus guide](https://user.cscs.ch/tools/containers/sarus/)). Sarus commands are similar to Docker or Singularity. It is recommended to create a sarus image in an interactive SLURM job using `srun --pty bash`. 

**Setup**

You should load `daint-gpu` or `daint-mc` before loading the `sarus` modulefile:

```shell
module load daint-gpu \# or daint-mc
module load sarus
```

Then you can pull a Docker image to a sarus image as follows:

```shell
sarus pull ghcr.io/i4ds/karabo-pipeline:latest
```

**Native MPI support (MPICH-based)**

Karabo >= `v0.21.0` supports [MPICH](https://www.mpich.org/)-based MPI processes that enable multi-node workflows on CSCS (or any other system which supports MPICH MPI). Our containers provide native MPI by hooking CSCS MPI into the container as follows:

```shell
srun -N16 -n16 -C gpu sarus run --mpi --mount=type=bind,source=<your_repo>,destination=/workspace ghcr.io/i4ds/karabo-pipeline:latest <mpi_application>
```

Here, an MPI application with 16 processes is launched with your repository mounted in the container (/workspace is the default working-directory). Make sure that you know how many processes are reasonable to run because it can rapidly sum up to a large number of nodehours.