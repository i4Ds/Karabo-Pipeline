# Create build container to not have copied filed in real container afterwards
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as build
ARG KARABO_TAG
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
RUN pip install jupyterlab ipykernel pytest && \
    python -m ipykernel install --user --name=karabo && \
    mkdir /workspace
WORKDIR /workspace
COPY --from=build Karabo-Pipeline/ repo/
RUN cp -r repo/karabo/examples /workspace/examples/ && \
    rm -rf repo/
