# Karabo Pipeline

# Sample Pipeline

This pipeline serves as the starting point for the SKA Digital Twin Pipeline, which is written in Python and set up in an interactive Jupyter Notebook environment. Two specific radio telescope packages are used:

- OSKAR: Responsible for the simulation of the sky and the telescope https://github.com/OxfordSKA/OSKAR
	- OSKAR telescope files telescope.tm are from https://github.com/OxfordSKA/OSKAR/releases -> Example Data
- RASCIL: Responsible for imaging https://gitlab.com/ska-telescope/external/rascil

# Docker

For easy usage use the docker already build docker image.
The image will start a Jupyter Lab server that you can then open in any browser on your host machine on [localhost:8888](localhost:8888).

Run this command to start a Jupyter Lab Server where your changes are NOT persistent between runs.
```shell
docker run -p 8888:8888 ghcr.io/i4ds/ska:main
```

Run this command to start a Jupyter Lab server where your changes in the Code are persistent between runs of the container.
```shell
docker run -p 8888:8888 -p 8787:8787 -v ska_pipeline_code:/home/jovyan/work/persistent ghcr.io/i4ds/ska:main
```

Now you can edit the code, run it and work with it, without installing any dependencies on your system.


# Installation

To install the pipeline on your local machine follow the [Instructions](doc/Installation.md). 
It might be easier to just use the docker file for fast testing.

