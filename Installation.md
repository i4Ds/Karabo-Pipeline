# Installation

To Install the package on your local system, follow the steps below.
The installation has been tested on Ubuntu 20.04 (x86_x64) and MacOS (Intel).

## 1. System prerequisites

Your system should have the following packages dependencies met

- C and C++ compiler (package build-essential on linux; on mac os installed via xcode)
- git
- git-lfs (for rascil)
- cmake
- python3 and pip3 (satisfied via anaconda)
- boost library (boost-python, boost numpy)
- casacore

Requirements can easily be installed on any debian based system with the apt package manager

````shell
apt update
apt install build-essential git git-lfs cmake libboost-all-dev casacore-dev libboost-numpy-dev libboost-python-dev
````

Install [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

Before installing all further depencies, make sure that the correct python and pip executables are selected in the current shell session.
If you do `conda activate <env>`, you will have the correct executables in your terminal.

It is advisable to first create a seperate environment, so run these.:

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

If you do not trust the script and would like to manually install the different depencies, you can do so by following the installation instructions in the links below.
Details about the installations are given in the documentation of the packages.

1. OSKAR installation: https://github.com/OxfordSKA/OSKAR & https://github.com/OxfordSKA/OSKAR/blob/master/python/README.md
2. RASCIL installation: https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html & copy the "data" directory from the pulled repository to the site-packages of the installation
