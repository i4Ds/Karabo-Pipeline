# Installation

To Install the package on your local system, follow the steps below.
The installation has been tested on Ubuntu 20.04 (x86_x64) and MacOS (Intel).

### Dependencies

Your system should have the following packages dependencies installed

- C and C++ compiler
- git
- git-lfs
- cmake
- python3 and pip3 (via anaconda or other python installation)
- boost-python
- casacore

The Pipeline uses the [OSKAR](https://github.com/OxfordSKA/OSKAR) and [RASCIL](https://ska-telescope.gitlab.io/external/rascil/index.html) packages.
The Installation procedure of the packages is not straight forward, therefore an installation scripts exist. See below.

### Instructions Linux (Ubuntu)

1. Install package dependencies

````shell
apt update
apt install build-essential git git-lfs cmake libboost-all-dev casacore-dev libboost-python-dev casacore-dev
````

2. Install [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

(Optional) 3. Create dedicated environment (environments can be python 3.7 up to python 3.9)

```shell
conda create -n pipeline_env python=3.8
conda activate pipeline_env
```

4. Install OSKAR and RASCIL

```shell
# do not run in sudo, this will install the pip packages as root (unless you want that)
./install.sh
```

(Optional if 3.) 5. Activate Environment as Kernel in Jupyter Lab

```shell
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=pipeline_env
```

### Instructions MacOS (Intel)

On MacOS the installation of the dependencies can be installed most easily with [Homebrew](https://brew.sh/).
The C and C++ compilers and git are installed via the XCode Command Line Tools.

```shell
brew tap ska-ska/tap #tap into repository where casacore is
brew update
brew install cmake git-lfs casacore boost
```

Then follow the same instructions as for Linux.

#### Some Remarks

Running under the new Apple Silicon is possible, however requires modification of the OSKAR Source Code.
If somebody wants to actually do that, please contact.
