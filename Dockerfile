FROM --platform=amd64 jupyter/base-notebook

ENV rascil_version=0.5.0

USER root

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libc6-dev \
    cmake \
    git \
    git-lfs \
    casacore-dev \
    wget \
    qt5-default \
    g++ \
    autotools-dev \
    libicu-dev \
    libbz2-dev \
    gfortran \
    software-properties-common \
    libboost-all-dev \
    curl && \
    apt-get clean

USER $NB_UID

RUN conda update conda && \
    conda install -c anaconda pip
RUN pip install dask-labextension

RUN conda create -n karabo python=3.7
SHELL ["conda", "run", "-n", "karabo", "/bin/bash", "-c"]
RUN conda install -c i4ds -c conda-forge karabo-pipeline
RUN conda install -c conda-forge python-casacore=3.4.0

RUN pip install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil
RUN mkdir rascil_data && \
    cd rascil_data && \
    curl https://ska-telescope.gitlab.io/external/rascil/rascil_data.tgz -o rascil_data.tgz && \
    tar zxf rascil_data.tgz && \
    cd data && \
    export RASCIL_DATA=`pwd`

RUN conda install ipykernel
RUN ipython kernel install --user --name=karabo

USER root

RUN mkdir /home/jovyan/work/persistent/

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" && \
    fix-permissions "${HOME}/work/persistent"

ENV JUPYTER_ENABLE_LAB=yes

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

USER $NB_UID
WORKDIR $HOME