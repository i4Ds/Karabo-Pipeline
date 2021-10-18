# Sample Pipeline

This pipeline serves as the starting point for the SKA Digial Twin Pipeline, which is written in Python and set up in an interactive Jupyter Notebook environment. Two specific radio telescope packages are used:

- OSKAR: Responsible for the simulation of the sky and the telescope https://github.com/OxfordSKA/OSKAR
	- OSKAR telescope files telescope.tm are from https://github.com/OxfordSKA/OSKAR/releases -> Example Data
- RASCIL: Responsible for imaging https://gitlab.com/ska-telescope/external/rascil

# Installation
## Prerequisites

The Installation has been tested on Ubuntu 20.04
Your system should have the following packages installed

- git
- git-lfs
- cmake
- python3 and pip3 (satisfied via anaconda)
- boost library (boost-python)
- casacore

Requirements can easily be installed on any debian based system with the apt package manager

````shell
apt update
apt install cmake git git-lfs libboost-all-dev casacore-dev
````

Install [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

Before installing all further depencies, make sure that the correct python and pip executables are selected in the current shell session.
If you do `conda activate <env>`, you will have the correct executables in your terminal.

It is advisable to first create a seperate environmen, so run these.:

```
conda create -n pipeline_env python=3.8
conda activate pipeline_env
```

The Pipeline uses the [OSKAR](https://github.com/OxfordSKA/OSKAR) and [RASCIL](https://ska-telescope.gitlab.io/external/rascil/index.html) packages.
The Installation procedure of the packages is not straight forward, therefore an installation scripts exist.

```shell
# run the installation script
# DO NOT RUN IN SUDO, this will install the pip packages as root (unless you want that)
# The installation script will use the in variables set in your current session.
# Meaning check which pip and python are used to execute the commands, so that the dependencies are installed correctly.
./install.sh
```

Or if you do not trust the script  would like to manually installe the different depencies, you can do so by following the installation instructions in the links below.
Details about the installations are given in the documentation of the packages.

1. OSKAR installation: https://github.com/OxfordSKA/OSKAR & https://github.com/OxfordSKA/OSKAR/blob/master/python/README.md
2. RASCIL installation: https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html & copy the "data" directory from the pulled repository to the site-packages of the installation

#Docker

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
