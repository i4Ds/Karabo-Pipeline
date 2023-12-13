FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
# build: user|test, KARABO_VERSION: version to install from anaconda.org in case build=user: `{major}.{minor}.{patch}` (no leading 'v')
ARG GIT_REV="main" BUILD="user" KARABO_VERSION=""
RUN apt-get update && apt-get install -y git gcc gfortran libarchive13 wget curl nano
ENV LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64" \
    PATH="/home/andromeda/miniconda3/bin:${PATH}" \
    IS_DOCKER_CONTAINER="true"
RUN useradd -ms /bin/bash andromeda
USER andromeda
WORKDIR /home/andromeda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.0-3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b && \
    conda init bash && \
    rm ~/miniconda.sh
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda create -y -n karabo && \
    echo "conda activate karabo" >> ~/.bashrc && \
    conda config --set auto_activate_base false
# change venv because libmamba solver lives in base and any serious environment update could f*** up the linked deps like `libarchive.so`
SHELL ["conda", "run", "-n", "karabo", "/bin/bash", "-c"]
RUN mkdir Karabo-Pipeline && \
    cd Karabo-Pipeline && \
    git init && \
    git remote add origin https://github.com/i4Ds/Karabo-Pipeline.git && \
    git fetch origin ${GIT_REV} && \
    git reset --hard ${GIT_REV} && \
    if [ "$BUILD" = "user" ] ; then \
    conda install -y -c i4ds -c conda-forge -c "nvidia/label/cuda-11.7.1" karabo-pipeline="$KARABO_VERSION"; \
    elif [ "$BUILD" = "test" ] ; then \
    conda env update -f="environment.yaml"; \
    pip install --no-deps "."; \
    else \
    exit 1; \
    fi && \
    mkdir ~/karabo && \
    cp -r "karabo/examples" ~/karabo/examples/ && \
    cd ".." && \
    rm -rf "Karabo-Pipeline/" && \
    pip install jupyterlab ipykernel pytest && \
    python -m ipykernel install --user --name=karabo

# The following steps installs mpich at a standard location to allow a native mpi-hook
# To make this work, conda additionally needs to link their mpi-installation in the virtual 
# environment to the standard-location (see issue #512). Be aware that the mpi-installation  
# and version is determined through `apt`. Therefore, to ensure abi-compatibility of mpi the 
# version installed using `apt` and the version specified in the environment-files must match.

# fetch mpich-version to have it consistent with it's installation from karabo
ARG MPICH_EVAL='echo $(conda list mpich -c | sed "s/.*mpich-\([0-9]\+\(\.[0-9]\+\)\+\)-.*/\1/")'
USER root
RUN MPICH_VERSION=$(eval $MPICH_EVAL) && \
    MPICH_VERSION_APT=$(echo "$MPICH_VERSION" | awk -F. '{print $1 "." $2 "-*"}') && \
    apt-get update && \
    apt-get install -y mpich=$MPICH_VERSION_APT
USER andromeda
RUN MPICH_VERSION=$(eval $MPICH_EVAL) && \
    conda install --force-reinstall -c conda-forge -y "mpich=${MPICH_VERSION}=external_*"
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "karabo"]