# Sample Pipeline

This pipeline serves as the starting point for the SKA Digial Twin Pipeline, which is written in Python and set up in an interactive Jupyter Notebook environment. Two specific radio telescope packages are used:

- OSKAR: Responsible for the simulation of the sky and the telescope https://github.com/OxfordSKA/OSKAR
	- OSKAR telescope files telescope.tm are from https://github.com/OxfordSKA/OSKAR/releases -> Example Data
- RASCIL: Responsible for imaging https://gitlab.com/ska-telescope/external/rascil

# Installation

To install the pipeline on your local machine follow the [Instructions](Installation.md). 
Note.: It might be easier to just use the docker file for fast testing.

# Docker

For easier use of the package, there are two docker files in the repository
```shell
# This dockerfile starts a jupyter server inside a docker file with all needed dependencies installed.
# When the container is running 
docker run -it $(docker build -f jupyter.Dockerfile .)
```

Or use the other docker file, which installs all dependencies and runs the `pipeline.py` (same code outside of notebook) file.

```shell
docker build -f pipeline.Dockerfile .
```

Then run the built image
```shell
docker run -it -p 8888:8888 <image-id>
```

Get access to the contrainer in a bash shell
```shell
docker exec -it <container-id> bash
```
