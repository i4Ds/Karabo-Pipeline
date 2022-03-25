# Installation

## Conda

We support the installation via the conda package manager.
Currently, python 3.8 is fully supported.

```shell
conda install -c i4ds -c conda-forge karabo-pipeline=0.2.0
```

Full command sequence for installation may look like this.

```shell
conda create -n karabo-env python=3.8
conda activate karabo-env
# karabo-pipeline
conda install -c i4ds -c conda-forge karabo-pipeline
```

#### Known Issues

Sometimes the installation of rascil fails due to the installation of pybdsf. If you have trouble with installing the pybdsf dependency, we recommend to check out their issue page for solutions.: https://github.com/lofar-astron/PyBDSF/issues


[Other installation methods](Installation_no_conda.md)

