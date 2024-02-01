# Installation (User)

## System Requirements
- Linux or Windows with WSL. For macOS we recommend you use [Docker](container.md), starting with version 0.18.1 of the image.
- 8GB RAM
- 10GB disk space
- GPU-acceleration requires proprietary nVidia drivers/CUDA >= 11

## Install Karabo
The following steps will install Karabo and its prerequisites (miniconda):

```shell
# install conda & solver
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda init bash
conda install -n base conda-libmamba-solver
# setup virtual environment
conda create -n karabo python=3.9
conda activate karabo
conda config --env --set solver libmamba
conda config --env --set channel_priority true
# install karabo
conda install -c nvidia/label/cuda-11.7.0 -c i4ds -c conda-forge karabo-pipeline
```

Karabo versions older than `v0.15.0` are deprecated and therefore installation will most likely fail. In addition, we do not support Karabo older than latest-minor version in case dependency resolving or online resources are outdated. Therefore, we strongly recommend using the latest version of Karabo. If an older version of Karabo is required, we strongly recommend using a [container](container.md), as the environment is fixed in a container. However, outdated online resources may still occur.

## Update to latest Karabo version
A Karabo installation can be updated the following way:

Note: Even though we care about not introducing API-breaking changes through different minor releases of Karabo, we don't guarantee it.

```
conda update -c nvidia/label/cuda-11.7.0 -c i4ds -c conda-forge karabo-pipeline
```

## Additional Notes and Troubleshooting
- Dont' install anything into the base environment except libraries which are supposed to live in there. If you accientally install packages there which are not supposed to be there, you might break some functionalities of your conda-installation.
- If you're using a system conda, it might be that you don't have access to a libmamba-solver, because the solver lives in the base environment, which belongs to root. In this case, you can ask your admin to install the solver, try an installation without the libmamba solver OR we recommend to just install conda into your home (which is the recommended solution).
- If you are using WSL and running a jupyter-notebook fails, you might have to set the path to the cuda libraries as follows:

```shell
conda env config vars set LD_LIBRARY_PATH=/usr/lib/wsl/lib
```
