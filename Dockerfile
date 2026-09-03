FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
# build: user|test, KARABO_VERSION: version to install from anaconda.org in case build=user: `{major}.{minor}.{patch}` (no leading 'v')
ARG GIT_REV="main" BUILD="user" KARABO_VERSION="" PYTHON_VERSION="3.12"
RUN apt-get update && apt-get install -y git gcc gfortran libarchive13t64 wget curl nano
ENV LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64" \
    PATH="/opt/conda/bin:${PATH}" \
    IS_DOCKER_CONTAINER="true"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py312_26.5.3-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda init && \
    rm ~/miniconda.sh
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda install -y -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda create -y -n karabo python=${PYTHON_VERSION}
# change venv because libmamba solver lives in base and any serious environment update could f*** up the linked deps like `libarchive.so`
SHELL ["conda", "run", "-n", "karabo", "/bin/bash", "-c"]
RUN mkdir Karabo-Pipeline && \
    cd Karabo-Pipeline && \
    git init && \
    git remote add origin https://github.com/i4Ds/Karabo-Pipeline.git && \
    git fetch && \
    git checkout ${GIT_REV} && \
    if [ "$BUILD" = "user" ] ; then \
    conda install -y -c "nvidia/label/cuda-12.9.1" -c i4ds -c conda-forge karabo-pipeline="$KARABO_VERSION"; \
    elif [ "$BUILD" = "test" ] ; then \
    conda env update -f="environment.yaml"; \
    python -m pip install --no-deps "."; \
    else \
    exit 1; \
    fi && \
    mkdir /workspace && \
    mkdir /workspace/karabo-examples && \
    cp -r karabo/examples/* /workspace/karabo-examples && \
    cd ".." && \
    rm -rf Karabo-Pipeline/ && \
    python -m pip install jupyterlab ipykernel pytest==7.4.2 && \
    python -m ipykernel install --user --name=karabo

# set bash-env accordingly for interactive and non-interactive shells for docker & singularity
RUN mkdir /opt/etc && \
    echo "conda activate karabo" >> ~/.bashrc && \
    cat ~/.bashrc | sed -n '/conda initialize/,/conda activate/p' > /opt/etc/conda_init_script
ENV BASH_ENV=/opt/etc/conda_init_script
RUN echo "source $BASH_ENV" >> /etc/bash.bashrc && \
    echo "source $BASH_ENV" >> /etc/profile

# link packaged mpich-version with ldconfig to enable mpi-hook (it also links everything else, but shouldn't be an issue)
RUN echo "$CONDA_PREFIX"/lib > /etc/ld.so.conf.d/conda.conf && \
    ldconfig

# Additional setup
WORKDIR /workspace
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "karabo"]