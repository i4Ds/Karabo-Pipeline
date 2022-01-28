# Installation

## Conda

We support the installation via the conda package manager.
Currently, python 3.7 is fully supported.

```shell
conda install -c i4ds -c conda-forge karabo-pipeline
```

With this one command you can use most of our package with no restrictions.
However, to use the imaging portion of it, you need to install rascil via pip aswell.

```shell
# rascil
conda install -c conda-forge python-casacore=3.4.0
pip install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil

# install data directory
mkdir rascil_data
cd rascil_data
curl https://ska-telescope.gitlab.io/external/rascil/rascil_data.tgz -o rascil_data.tgz
tar zxf rascil_data.tgz
cd data
export RASCIL_DATA=`pwd`
```

You may also use other isntallation methods of rascil see.: https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html

Full command sequence for installation may look like this.

```shell
conda create -n karabo-env python=3.7
conda activate karabo-env
# karabo-pipeline (inlcuding oskar)
conda install -c i4ds -c conda-forge karabo-pipeline
# rascil
pip install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil
mkdir rascil_data
cd rascil_data
curl https://ska-telescope.gitlab.io/external/rascil/rascil_data.tgz -o rascil_data.tgz
tar zxf rascil_data.tgz
cd data
export RASCIL_DATA=`pwd`
cd ../..
```

#### Known Issues

Sometimes the installation of rascil fails due to the installation of pybdsf. If you have trouble with installing the pybdsf dependency, we recommend to check out their issue page for solutions.: https://github.com/lofar-astron/PyBDSF/issues


[Other installation methods](Installation_no_conda.md)

