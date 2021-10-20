FROM ubuntu:latest

ARG USER=dtwin
ENV HOME=/home/${USER}
RUN mkdir ${HOME}
WORKDIR ${HOME}

ENV PATH="${HOME}/miniconda3/bin:${PATH}"
ARG PATH="${HOME}/miniconda3/bin:${PATH}"
ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && \
    apt-get install -y \
    build-essential \ 
    cmake \
    git \
    git-lfs \
    libboost-all-dev \
    casacore-dev \
    wget


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda update -y conda
RUN conda install -c anaconda pip
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge matplotlib 
RUN conda install -c anaconda astropy

EXPOSE 8888

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
COPY install.sh install.sh
RUN ./install.sh

RUN mkdir pipeline
COPY . pipeline/.

ENTRYPOINT [ "jupyter" , "lab", "--ip=0.0.0.0", "--port=8888" , "--allow-root", "--notebook-dir=pipeline" ]

#ENTRYPOINT [ "python3", "pipeline.py" ]
