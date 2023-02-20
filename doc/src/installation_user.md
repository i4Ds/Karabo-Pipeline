# Installation (User)

## System Requirements
- Linux or Windows with WSL. For macOS we recommend you use [Docker](container.html).
- 8GB RAM
- 10GB disk space
- GPU-acceleration requires proprietary nVidia drivers/CUDA

## Install Karabo
The following steps will install Karabo and its prerequisites (miniconda, libmamba):
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh
bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
conda init bash
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba
conda update -y -n base -c defaults conda
conda create -y -n karabo-env python=3.9
conda activate karabo-env
conda install -y -c i4ds -c conda-forge -c nvidia/label/cuda-11.7.0 karabo-pipeline
conda clean --all -y
```

## Update to the current Karabo version
A Karabo installation can be updated the following way:
```
conda update -y -c i4ds -c conda-forge -c nvidia/label/cuda-11.7.0 karabo-pipeline
conda clean --all -y
```

## Additional Notes and Troubleshooting
- If the base environment was updated, *libmamba* might fail to install. In that case, reset conda to version 22 using `conda install --rev 0 --name base` or you can try installing Karabo without *libmamba*. Using *libmamba* is not strictly required, but strongly recommended, because it makes the installation much faster and more reliable.
- You can install miniconda into a different path, use ```bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b -p YourDesiredPath``` instead
- If you are using WSL and running a jupyter-notebook fails, you might have to set the path to the cuda libraries as follows:

```shell
conda env config vars set LD_LIBRARY_PATH=/usr/lib/wsl/lib
```
