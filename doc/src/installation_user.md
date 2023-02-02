# Installation (User)

## Requirements

### Operating System
- Linux 
- Windows 10+ (21H1+) with the Hypervisor based Windows Subsystem for Linux (WSL)

### Memory Requirements
- 8GB RAM

### Disk Space Requirements
#### Recommended - For installation
-  10GB

#### Cleaned Installation Size (description below)
-  Karabo: ~2.5GB
-  Miniconda with Libmamba: ~1GB

### Tools
- Miniconda / Anaconda

### CUDA (if you want to use your Nvidia GPU)
- The proprietary Nvidia driver

## Install Conda

We have noticed that dependency-resolving with conda's native solver can take a while. Therefore, we recommend installing Miniconda and use the libmamba solver for installation.

1. Go to [MiniConda install guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) and follow the steps.

### Install libmamba with the following steps:

Update your base environment
- `conda update -n base -c defaults conda`

Install the LibMamba Solver
- `conda install -n base conda-libmamba-solver`

Libmamba can be set as the standard solver for conda. If you do not wish to have it as standard but want to use it in some workloads, add the following parameter at the end of your conda install statement `--experimental-solver=libmamba`

Example: `conda install python=3.10 --experimental-solver=libmamba`

If you want to use it everywhere, use the following statement to declare it as the standard solver
- `conda config --set solver libmamba`

## Karabo Pipeline installation with Conda

We support the installation via the conda package manager.
Currently, Python 3.9 is fully supported.

Python 3.10 will follow in the future - if you need it right now, contact us :) 

Full command sequence for installation may look like this.

```shell
conda create -n karabo-env python=3.9
conda activate karabo-env
# karabo-pipeline
conda install -c i4ds -c conda-forge -c nvidia/label/cuda-11.7.0 karabo-pipeline
```

Run this command to free disk space after the installation. This will remove all temporary and cached packages - after this you get the installation size described in the requirements.
- `conda clean --all -y`

## MacOS Support

MacOS is not supported natively. Even though macOS is similar to Linux, the native libraries are not (ARM based macs) or very hard (x86 macs) to build on macOS. 

Please use Docker with Ubuntu to run Karabo on macOS. Prepared Docker files are in KaraboRoot/docker. This also works on ARM based macs (M1, M2, ..., Mx)

##  Known Issues

### Oskar
Sometimes Oskar is not correctly installed. 

Please also have a look at [Other installation methods](installation_no_conda.md) to see how to install Oskar manually.

### Canceled future for execute_request message before replies were done
This happens certain IDEs on WSL when using a jupyter notebook. To fix this, you need to export a variable manually: 
`LD_LIBRARY_PATH` has to include `/usr/lib/wsl/lib`. 
In the beginning of the notebook, you can add the variable (before importing karabo) as follows:

```python
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib'
import karabo
```

### undefined symbol: H5Pset_*
Sometimes, the package causes an error similar to `undefined symbol: H5Pset_fapl_ros3`. 

Downgrading `h5py` to 3.1 with the following command fixes it:

```shell
pip install h5py==3.1
```

### UnsatisfiableError: The following specifications were *

```python
UnsatisfiableError: The following specifications were found to be incompatible with each other:
Output in format: Requested package -> Available versionsThe following specifications were found to be incompatible with your system:

feature:/linux-64::__glibc==2.31=0
feature:|@/linux-64::__glibc==2.31=0

	Your installed version is: 2.3
```

This is usually fixable when fixing the python version to 3.9 when creating the environment.
