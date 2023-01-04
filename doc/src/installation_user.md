# Installation (User)

## Requirements

### Operating System
- Linux 
- Windows 10+ (21H1+) with the Hypervisor based Windows Subsystem for Linux (WSL)

### Tools
- Conda

### CUDA (if you want to use your Nvidia GPU)
- The proprietary Nvidia driver

## Install Conda

1. Go to https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
2. Follow the instructions for Miniconda until step 6. 

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

## MacOS Support

MacOS is not supported natively. Even though macOS is similar to Linux, the native libraries are not (ARM based macs) or very hard (x86 macs) to build on macOS. 

Please use Docker with Ubuntu to run Karabo on macOS. Prepared Docker files are in KaraboRoot/docker. This also works on ARM based macs (M1, M2, ..., Mx)

##  Known Issues

### Oskar
Sometimes Oskar is not correctly installed. 

Please also have a look at [Other installation methods](installation_no_conda.md) to see how to install Oskar manually.

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
