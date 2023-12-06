# for copying example-notebooks & in case env is installed through `environment.yaml`
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as karabo-repo
ARG GIT_REV
RUN apt-get update && apt-get install -y git && \
    mkdir Karabo-Pipeline && \
    cd Karabo-Pipeline && \
    git init && \
    git remote add origin https://github.com/i4Ds/Karabo-Pipeline.git && \
    git fetch origin ${GIT_REV} && \
    git reset --hard ${GIT_REV}

# main build
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
# build: user|test, KARABO_VERSION: version to install from anaconda.org in case build=user: `{major}.{minor}.{patch}` (no leading 'v')
ARG BUILD=user KARABO_VERSION=""
RUN apt-get update && apt-get install -y git gcc gfortran libarchive13 wget curl nano
ENV LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64" \
    PATH="/opt/conda/bin:${PATH}" \
    CONDA_PREFIX="/opt/conda" \
    IS_DOCKER_CONTAINER="true"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.0-3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_PREFIX} && \
    conda init
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda create -y -n karabo
# change venv because libmamba solver lives in base and any serious environment update could f*** up the linked deps like `libarchive.so`
SHELL ["conda", "run", "-n", "karabo", "/bin/bash", "-c"]
COPY --from=karabo-repo Karabo-Pipeline/ repo/
RUN if [ "$BUILD" = "user" ] ; then \
    conda install -y -c i4ds -c conda-forge -c "nvidia/label/cuda-11.7.1" karabo-pipeline="$KARABO_VERSION"; \
    elif [ "$BUILD" = "test" ] ; then \
    cd "repo"; \
    conda env update -f="environment.yaml"; \
    pip install --no-deps "."; \
    cd ".." ; \
    else \
    exit 1; \
    fi && \
    echo "conda activate karabo" >> ~/.bashrc  && \
    mkdir /workspace && \
    cp -r "repo/karabo/examples" "/workspace/examples/" && \
    rm -rf "repo/" && \
    pip install jupyterlab ipykernel pytest && \
    python -m ipykernel install --user --name=karabo
WORKDIR /workspace
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "karabo"]

# The following steps are needed if an mpi-hook through sarus must be allowed.
# However, because this steps takes more than 2 hours, and the github-runners run out
# of memory, these steps are outcommented at the time-beeing.

# # fetch mpich-version to have it consistent with it's installation from karabo
# ARG MPICH_EVAL='echo $(conda list mpich -c | sed "s/.*mpich-\([0-9]\+\(\.[0-9]\+\)\+\)-.*/\1/")'
# # install mpich on standard location to enable mpi-hook (may take a while)
# RUN MPICH_VERSION=$(eval $MPICH_EVAL) && \
#     mkdir -p /tmp/mpich-build && \
#     cd /tmp/mpich-build && \
#     wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
#     tar xvzf mpich-${MPICH_VERSION}.tar.gz && \
#     cd mpich-${MPICH_VERSION}  && \
#     ./configure --enable-fast=all,O3 --prefix=/usr --with-cuda=/usr/local/cuda && \
#     make -j4 && \
#     make install && \
#     ldconfig && \
#     cp -p /tmp/mpich-build/mpich-${MPICH_VERSION}/examples/cpi /usr/bin/ && \
#     cd / && \
#     rm -rf /tmp/mpich-build
# # replace mpi with dummy-install (see issue #512)
# RUN MPICH_VERSION=$(eval $MPICH_EVAL) && \
#     conda install --force-reinstall -c conda-forge -y "mpich=${MPICH_VERSION}=external_*"