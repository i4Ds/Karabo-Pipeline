# Containers

## Docker Images

We provide for the karabo-pipeline [Docker images](https://www.docker.com/resources/what-container/#:~:text=A%20Docker%20container%20image%20is,tools%2C%20system%20libraries%20and%20settings.) which are hosted by the [ghcr.io](https://github.com/features/packages) registry. An overview of all available images is [here](https://github.com/i4ds/Karabo-Pipeline/pkgs/container/karabo-pipeline), if a specific version and not simply `latest` is desired. Starting from `karabo@v0.15.0`, all versions should be available. Provided you have docker, the image can be installed as follows:

```shell
docker pull ghcr.io/i4ds/karabo-pipeline:latest
```

Docker images have the advantage that the packages needed for karabo-pipeline are already pre-installed and you can usually run them on other operating systems. In addition, Docker images can easily create singularity containers (see [Singularity Container](#singularity-container)), which are often used in HPC clusters.

## Singularity Container

Singularity containers are often standard on HPC clusters, which do not require special permissions (unlike Docker).
We do not provide ready-made [Singularity containers](https://sylabs.io/). However, they can be easily created from Docker images with the following command (may take a while):

```shell
singularity pull https://ghcr.io/i4ds/karabo-pipeline:latest
```

How to use Singularity containers can be seen in the [Singularity documentation](https://docs.sylabs.io/guides/3.1/user-guide/cli.html).

## Docker Container

What is possible with Docker is far too extensive to describe here. We refer to the official [Docker reference](https://docs.docker.com/reference/) for this. We only show here a minimal example of how Docker could be used, so you can use a [Jupyter Notebook](https://jupyter.org/) with sample code and working Karabo environment.

```shell
docker run -it --rm -p 8888:8888 ghcr.io/i4ds/karabo-pipeline:latest
```

This starts the Docker container of the image interactively, where we have port 8888 forwarded here. After that, we start the jupyter service in the container with the following command:

```shell
jupyter lab --ip 0.0.0.0 --no-browser --port=8888 --allow-root
```

This will start the server on the same port we forwarded. Then copy the url which is given at the bottom and replace `hostname` with `localhost` and open it in the browser.