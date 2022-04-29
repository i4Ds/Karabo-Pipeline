[![Test Software](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yaml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yaml)
[![Build Docs](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-docs.yaml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-docs.yaml)
[![Build Conda](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/conda-build.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/conda-build.yml)
[![Build Docker CLI](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-cli-docker-image.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-cli-docker-image.yml)
[![Build Docker Jupyter](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-jupyter-docker-image.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-jupyter-docker-image.yml)

# Karabo Pipeline

This pipeline serves as the starting point for the SKA Digital Twin Pipeline, which is written in Python and set up in an interactive Jupyter Notebook environment. Two specific radio telescope packages are used:

- OSKAR: Responsible for the simulation of the sky and the telescope https://github.com/OxfordSKA/OSKAR
	- OSKAR telescope files telescope.tm are from https://github.com/OxfordSKA/OSKAR/releases -> Example Data
- RASCIL: Responsible for imaging https://gitlab.com/ska-telescope/external/rascil

## Local Installation
#### Conda
```shell
conda create -n karabo python
conda activate karabo
conda install -c i4ds -c conda-forge karabo-pipeline=0.3.0
```

[Installation](doc/src/Installation.md)

## Containers

Complete Jupyter Environment in a Docker container

```shell
docker run -p 8888:8888 -v ska_pipeline_code:/home/jovyan/work/persistent ghcr.io/i4ds/karabo-pipeline:jupyter
```

[Containers](doc/src/Container.md)
