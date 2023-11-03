# This Dockerfile is designed for CI/CD purpose (not mounting your repo and work with the container)
# If you want to mount your repo, you more or less just need the first stage, then install your own 
# deps from the mounted repo, and don't kill the container once you've installed your environment.

# first stage is to have a more or less consistent base-image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as karabo-base
RUN apt-get update && apt-get install -y git libarchive13 wget curl nano
ENV PATH="/opt/conda/bin:${PATH}" CONDA_PREFIX="/opt/conda" IS_DOCKER_CONTAINER="true"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.0-3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_PREFIX}
RUN conda init
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba
WORKDIR /workspace

#second stage is to build an image which is changing very often (e.g. env installation for CI-jobs)
# because dev-image is used for ci-purpose and it's mpi-implementation and version is not known prior,
# we don't compile the mpi from source here because it takes just too long. This makes an mpi-hook not possible.
FROM karabo-base
# redefine envs because they're just scoped per build-stage
ENV PATH="/opt/conda/bin:${PATH}" CONDA_PREFIX="/opt/conda" IS_DOCKER_CONTAINER="true"
# KARABO_TAG can be a branch, tag or commit (set from CI-job to get specific repo checkout)
ARG KARABO_TAG
# keep ADD instead of RUN because ADD always evaluates it's content to decide whether to use the cache or not
ADD https://github.com/i4Ds/Karabo-Pipeline.git#${KARABO_TAG} Karabo-Pipeline/
WORKDIR /workspace/Karabo-Pipeline
# note that installation like this has several assumptions about the used files like:
# conda-channel definition & not naming env in `environment.yaml`, dev-optional dep in pyproject.toml
RUN conda env update -f=environment.yaml && \
    pip install -e ".[dev]"