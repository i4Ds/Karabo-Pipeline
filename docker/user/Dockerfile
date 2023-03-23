# Create build container to not have copied filed in real container afterwards
FROM --platform=amd64 ghcr.io/i4ds/continuumio/miniconda3:4.12.0 as build
ARG KARABO_TAG
RUN apt-get update && apt-get install -y git
# RUN git clone --branch ${KARABO_TAG} --depth=1 https://github.com/i4Ds/Karabo-Pipeline.git
RUN git clone --branch LG_298_docker --depth=1 https://github.com/i4Ds/Karabo-Pipeline.git

FROM --platform=amd64 ghcr.io/i4ds/continuumio/miniconda3:4.12.0
ARG KARABO_TAG
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN apt-get update && apt-get install -y curl && apt-get autoclean && rm -rf /var/lib/apt/lists/*
RUN conda update -y conda && \
    conda clean --all --yes && \
    conda install mamba -y -c conda-forge
# REL_LABEL=rel_v{MAJOR}.{MINOR}, assumption is that `KARABO_TAG` has a leading "v", e.g. v0.15.0
RUN REL_LABEL=rel_$(echo $KARABO_TAG | cut -d '.' -f 1-2) && \
    mamba install -y -c "i4ds/label/$REL_LABEL" -c conda-forge -c nvidia/label/cuda-11.7.0 karabo-pipeline="${KARABO_TAG:1}"
RUN pip install jupyterlab ipykernel
RUN python -m ipykernel install --user --name=karabo
RUN mkdir /workspace
WORKDIR /workspace
COPY --from=build Karabo-Pipeline/ repo/
RUN cp -r repo/karabo/examples /workspace/examples/

# test karabo
# repo/ ensures that karabo is from site-packages and not from local
# tmp because tests produce pemanent files
# comment out ONLY the next layer to disable testing
# RUN mkdir tmp && \
#     cp -r repo/ tmp/repo && \
#     cd tmp && \
#     if [[ -f "repo/requirements.txt" ]] ; then pip install -r repo/requirements.txt ; else pip install pytest ; fi && \
#     python -c "from karabo.util.testing import run_tests; run_tests()" && \
#     cd .. && \
#     rm -rf tmp/
RUN rm -rf repo/
