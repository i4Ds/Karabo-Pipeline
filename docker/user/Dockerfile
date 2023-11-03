# Create build container to not have copied filed in real container afterwards
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as build
ARG KARABO_TAG
RUN apt-get update && apt-get install -y git
ADD https://github.com/i4Ds/Karabo-Pipeline.git#${KARABO_TAG} Karabo-Pipeline/

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install -y gcc gfortran libarchive13 wget curl nano
ARG KARABO_TAG
ENV PATH="/opt/conda/bin:${PATH}" CONDA_PREFIX="/opt/conda" IS_DOCKER_CONTAINER="true"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.0-3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_PREFIX}
RUN conda init
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
# install mpich on standard location (needed for mpi-hook)
RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba
RUN conda install -y -c i4ds -c conda-forge -c "nvidia/label/cuda-11.7.1" karabo-pipeline="${KARABO_TAG:1}"
# fetch mpich-version to have it consistent with it's installation from karabo
ARG MPICH_EVAL='echo $(conda list mpich -c | sed "s/.*mpich-\([0-9]\+\(\.[0-9]\+\)\+\)-.*/\1/")'
RUN MPICH_VERSION=$(eval $MPICH_EVAL) && \
    mkdir -p /tmp/mpich-build && \
    cd /tmp/mpich-build && \
    wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar xvzf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION}  && \
    ./configure --enable-fast=all,O3 --prefix=/usr --with-cuda=/usr/local/cuda && \
    make -j4 && \
    make install && \
    ldconfig && \
    cp -p /tmp/mpich-build/mpich-${MPICH_VERSION}/examples/cpi /usr/bin/ && \
    cd / && \
    rm -rf /tmp/mpich-build
# replace mpi with dummy-install (see issue #512)
RUN MPICH_VERSION=$(eval $MPICH_EVAL) && \
    conda install --force-reinstall -c conda-forge -y "mpich=${MPICH_VERSION}=external_*"
RUN pip install jupyterlab ipykernel pytest && \
    python -m ipykernel install --user --name=karabo && \
    mkdir /workspace
WORKDIR /workspace
COPY --from=build Karabo-Pipeline/ repo/
RUN cp -r repo/karabo/examples /workspace/examples/ && \
    rm -rf repo/
