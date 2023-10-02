# This Dockerfile is for developing purpose only
# Make sure to call `docker build` at the root of the repo to have access to the required files

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install -y libarchive13 wget curl nano
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.0-3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=/opt/conda/bin:$PATH CONDA_PREFIX=/opt/conda
RUN conda init
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda update -y conda && \
    conda install mamba -y -c conda-forge
COPY environment.yaml requirements.txt ./
RUN mamba env update --file environment.yaml && \
    pip install -r requirements.txt && \
    pip install jupyterlab ipykernel && \
    python -m ipykernel install --user --name=karabo && \
    rm environment.yaml requirements.txt
RUN mkdir /workspace
WORKDIR /workspace