# Create build container to not have copied filed in real container afterwards
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as build
ARG KARABO_TAG
ARG IS_DOCKER_CONTAINER=true
RUN apt-get update && apt-get install -y git
RUN git clone --branch ${KARABO_TAG} --depth=1 https://github.com/i4Ds/Karabo-Pipeline.git

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG KARABO_TAG
RUN apt-get update && apt-get install -y libarchive13 wget curl nano
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.0-3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=/opt/conda/bin:$PATH CONDA_PREFIX=/opt/conda
RUN conda init
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda update -y conda && \
    conda install mamba -y -c conda-forge
RUN mamba install -y -c i4ds -c conda-forge -c "nvidia/label/cuda-11.7.1" karabo-pipeline="${KARABO_TAG:1}"
ARG MPICH_VERSION="4.1.2"
ARG MPICH_CONFIGURE_OPTIONS="--enable-fast=all,O3 --prefix=/usr --disable-fortran --with-cuda=/usr/local/cuda"
ARG MPICH_MAKE_OPTIONS="-j4"
# install mpich on standard location (needed for mpi-hook)
RUN mkdir -p /tmp/mpich-build \
    && cd /tmp/mpich-build \
    && wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz \
    && tar xvzf mpich-${MPICH_VERSION}.tar.gz \
    && cd mpich-${MPICH_VERSION}  \
    && ./configure ${MPICH_CONFIGURE_OPTIONS} \
    && make ${MPICH_MAKE_OPTIONS} \
    && make install \
    && ldconfig \
    && cp -p /tmp/mpich-build/mpich-${MPICH_VERSION}/examples/cpi /usr/bin/ \
    && cd / \
    && rm -rf /tmp/mpich-build
# replace openmpi with mpich
RUN conda remove --force-remove -y openmpi mpi && \
    conda install -c conda-forge -y "mpich=${MPICH_VERSION}=external_*"
RUN pip install jupyterlab ipykernel pytest && \
    python -m ipykernel install --user --name=karabo && \
    mkdir /workspace
WORKDIR /workspace
COPY --from=build Karabo-Pipeline/ repo/
RUN cp -r repo/karabo/examples /workspace/examples/ && \
    rm -rf repo/
