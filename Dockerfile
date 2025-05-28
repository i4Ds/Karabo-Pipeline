FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
# build: user|test, KARABO_VERSION: version to install from anaconda.org in case build=user: `{major}.{minor}.{patch}` (no leading 'v')
ARG GIT_REV="upgrade_python" BUILD="user" KARABO_VERSION=""
RUN apt-get update && apt-get install -y git gcc gfortran libarchive13 wget curl nano

ENV LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64" \
    PATH="/opt/conda/bin:${PATH}" \
    IS_DOCKER_CONTAINER="true"

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda init && \
    rm ~/miniconda.sh

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda install -y -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda create -y -n karabo python=3.10

# change venv because libmamba solver lives in base and any serious environment update could f*** up the linked deps like `libarchive.so`
SHELL ["conda", "run", "-n", "karabo", "/bin/bash", "-c"]
RUN mkdir Karabo-Pipeline && \
    cd Karabo-Pipeline && \
    git init && \
    git remote add origin https://github.com/i4Ds/Karabo-Pipeline.git && \
    git fetch && \
    git checkout ${GIT_REV} && \
    if [ "$BUILD" = "user" ] ; then \
      conda install -y -c i4ds -c i4ds/label/dev -c conda-forge -c "nvidia/label/cuda-11.7.1" karabo-pipeline="$KARABO_VERSION"; \
    elif [ "$BUILD" = "test" ] ; then \
      conda env update -f="environment.yaml"; \
      pip install --no-deps "."; \
    else \
      exit 1; \
    fi && \
    mkdir /workspace && \
    mkdir /workspace/karabo-examples && \
    cp -r karabo/examples/* /workspace/karabo-examples && \
    cd ".." && \
    rm -rf Karabo-Pipeline/ && \
    pip install --no-cache-dir jupyterlab ipykernel pytest && \
    python -m ipykernel install --user --name=karabo

# PATCH healpy and tools21cm after install to fix scipy>=1.14 incompat
# NOTE: healpy and tools21cm still import deprecated APIs (trapz/quadrature) from scipy.integrate
# scipy >=1.14 removed these functions → leads to ImportError
# Temporary fix: monkey-patch healpy/tools21cm post-install
# Long-term: remove patch and pin healpy once it properly restricts scipy
RUN sed -i 's/from scipy.integrate import trapz/import numpy as np\ntrapz = np.trapz/' \
    /opt/conda/envs/karabo/lib/python3.10/site-packages/healpy/sphtfunc.py && \
    sed -i 's/from scipy.integrate import quadrature/import warnings\nwarnings.warn("quadrature removed in scipy>=1.14; replace or downgrade scipy", DeprecationWarning)\nquadrature = None/' \
    /opt/conda/envs/karabo/lib/python3.10/site-packages/tools21cm/cosmology.py && \
    sed -i 's/from scipy.integrate import quadrature/import warnings\nwarnings.warn("quadrature removed in scipy>=1.14; replace or downgrade scipy", DeprecationWarning)\nquadrature = None/' \
    /opt/conda/envs/karabo/lib/python3.10/site-packages/tools21cm/foreground_model.py

# Clean up conda cache to reduce image size
RUN conda clean -a -y

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