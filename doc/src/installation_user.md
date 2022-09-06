# Installation (User)

## Conda

We support the installation via the conda package manager.
Currently, python 3.9.13 is fully supported.

Full command sequence for installation may look like this.

```shell
conda create -n karabo-env python=3.9.13
conda activate karabo-env
# karabo-pipeline
conda install -c i4ds -c conda-forge karabo-pipeline
```

# Known Issues
## Rascil
Sometimes the installation of rascil fails due to the installation of pybdsf. If you have trouble with installing the pybdsf dependency, we recommend to check out their issue page for solutions.: https://github.com/lofar-astron/PyBDSF/issues


[Other installation methods](installation_no_conda.md)

## Rascil

## undefined symbol: H5Pset_*
Sometimes, the package causes an error similar to `undefined symbol: H5Pset_fapl_ros3`. 

Downgrading `h5py` to 3.1 with the following command fixes it:

```shell
pip install h5py==3.1
```

## UnsatisfiableError: The following specifications were *

```python
UnsatisfiableError: The following specifications were found to be incompatible with each other:
Output in format: Requested package -> Available versionsThe following specifications were found to be incompatible with your system:

feature:/linux-64::__glibc==2.31=0
feature:|@/linux-64::__glibc==2.31=0

	Your installed version is: 2.3
```

This is usually fixable when fixing the python version to 3.9.13 when creating the enviorment.