# Installation (User)

## System Requirements
- Linux or Windows with WSL. For macOS we recommend you use [Docker](container.html).
- 8GB RAM
- 10GB disk space
- GPU-acceleration requires proprietary nVidia drivers/CUDA >= 11.7

## Install Karabo
The following steps will install Karabo and its prerequisites (miniconda):

Karabo dependency resolving used to be difficult. The current solution requires to point at a specific release label `-c "i4ds/label/{REL_LABEL}" ` in the `conda install` command. Karabo releases follow the `MAJOR.MINOR.PATCH` versioning. Just replace {REL_LABEL} with the `rel_v{MAJOR}.{MINOR}` of the Karabo version you intend to install. E.g. Karabo `v0.15.0` gets the label `rel_v0.15`. An overview of the Karabo-tags is on our [GitHub Karabo-Pipeline](https://github.com/i4Ds/Karabo-Pipeline/tags).

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
conda install -y -c "i4ds/label/{REL_LABEL}" -c conda-forge -c nvidia/label/cuda-11.7.0 karabo-pipeline
conda clean --all -y
```

Karabo releases older than `v0.15.0` are deprecated and therefore we don't guarantee a successful installation. They don't follow the above depicted label convention for `conda install`.

## Update to the current Karabo version
A Karabo installation can be updated the following way:
```
conda update -y -c "i4ds/label/{REL_LABEL}" -c conda-forge -c nvidia/label/cuda-11.7.0 karabo-pipeline
conda clean --all -y
```

## Additional Notes and Troubleshooting
- If the base environment was updated, *libmamba* might fail to install. In that case, reset conda to version 22 using `conda install --rev 0 --name base` or you can try installing Karabo without *libmamba*. Using *libmamba* is not strictly required, but strongly recommended, because it should make the installation much faster and more reliable.
- You can install miniconda into a different path, use ```bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b -p YourDesiredPath``` instead
