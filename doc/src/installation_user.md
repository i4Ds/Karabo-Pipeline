# Installation for Users

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

## Next steps: run the examples
Now that you have a working installation, 
you can try out the [examples](examples/examples.md). For this, start a python session 
from the command line 
or, better, use a jupyter notebook. 
Jupyter is not installed by default by Karabo, but you
can install it with:
```
conda install jupyter
```
and call it with 
```
jupyter notebook
```
It can get a bit tricky if you are on a cloud environment, such as the amazon cloud. 
There are 
[instructions](https://towardsdatascience.com/setting-up-and-using-jupyter-notebooks-on-aws-61a9648db6c5) on how to use jupyter with an EC2 instances that should help setting
it up correctly. 

## Additional Notes and Troubleshooting
- If the base environment was updated, *libmamba* might fail to install. In that case, reset conda to version 22 using `conda install --rev 0 --name base` or you can try installing Karabo without *libmamba*. Using *libmamba* is not strictly required, but strongly recommended, because it makes the installation much faster and more reliable.
- You can install miniconda into a different path, use ```bash Miniconda3-py39_22.11.1-1-Linux-x86_64.sh -b -p YourDesiredPath``` instead
