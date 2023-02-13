# Containers

## Images

There are two docker images available for using the karabo-pipeline without any installation inside a safe environment

- karabo-jupyter: Starting this image provides a full jupyter lab environment with the pipeline pre-installed. Useful for playing around with the pipeline with no commitment and hassle on your own machine.
- karabo-cli: Use only the CLI portion of the pipeline with this image. Useful for running on HPC Machines, where you can only use SSH to control it.


## Singularity Images

! Coming Soon

## Docker Images

### Jupyter Image


```shell
docker pull ghcr.io/i4ds/karabo-pipeline:jupyter
```

Run the image, with the ``-p 8888:8888`` to expose Jupyter lab to your computer, so you can use it in your browser. And add `-v` to mount a volume so the code you create inside the container is saved on your host.

```shell
docker run -p 8888:8888 -v ska_pipeline_code:/home/jovyan/work/persistent ghcr.io/i4ds/karabo-pipeline:jupyter
```

Inside the Image:

Choose the Karabo Kernel to run your Karabo pipeline.

#### Compose

If you are familiar with docker-compose, you can also start run it with this compose file. Save this into a file called ``compose.yaml`` and start the Jupyter Lab Server with ``docker-compose -f compose.yaml up``

```yaml

version: '3'
services:
  jupyter-lab-docker:
    image: ghcr.io/i4ds/karabo-jupyter:main
    volumes:
      - src:/home/jovyan/work/persistent
    ports:
      - "8888:8888" #jupyter lab port
      - "8787:8787" #dask port
volumes:
  pipeline:
```

### CLI (Command Line Interface) Image

```shell
docker pull ghcr.io/i4ds/karabo-pipeline:cli
```

```shell
docker run -it ghcr.io/i4ds/karabo-pipeline:cli
```
