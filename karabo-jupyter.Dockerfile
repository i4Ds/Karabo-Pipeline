# syntax=docker/dockerfile:1.6
FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2

# Install Karabo via conda
USER root

# TODO: python 3.10

# Create new environment with Python 3.9 and install dependencies using libmamba solver (faster)
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    conda install -y -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda create -y -n karabo python=3.9 && \
    conda install -y -n karabo -c i4ds -c conda-forge -c "nvidia/label/cuda-11.7.1" karabo-pipeline pyuvdata && \
    # Ensure OSKAR can find a CUDA 11 runtime (libcudart.so.11.0)
    conda install -y -n karabo -c "nvidia/label/cuda-11.7.1" cuda-cudart=11.7.* && \
    # Create a separate env holding CUDA 10.2 runtime to supply libcufft.so.10 for OSKAR
    conda create -y -n cuda10 -c defaults cudatoolkit=10.2 && \
    conda run -n karabo pip install --no-cache-dir ipykernel pytest && \
    printf "/opt/conda/envs/karabo/lib\n/opt/conda/envs/cuda10/lib\n/opt/conda/lib\n" > /etc/ld.so.conf.d/conda.conf && \
    ldconfig && \
    # Symlink required CUDA 10 libs into karabo env so OSKAR finds them via RPATH
    bash -lc 'set -euo pipefail; for so in libcufft.so.10 libcudart.so.10 libcurand.so.10; do \
      for f in /opt/conda/envs/cuda10/lib/${so}*; do \
        if [ -e "$f" ]; then ln -sf "$f" /opt/conda/envs/karabo/lib/; fi; \
      done; \
    done' && \
    ldconfig && \
    fix-permissions /home/$NB_USER

# Workaround JupyterLab extension-manager crash with httpx>=0.28 (base runs Jupyter)
RUN conda run -n base python -m pip install --no-cache-dir "httpx<0.28"

# Make the karabo env's python first on PATH for non-interactive commands
ENV PATH="/opt/conda/envs/karabo/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/envs/karabo/lib:/opt/conda/envs/cuda10/lib:/opt/conda/lib"
ENV OSKAR_CUDA_LIB_PATH="/opt/conda/envs/cuda10/lib"

# Copy repository into the image to run tests against it
COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/Karabo-Pipeline
RUN fix-permissions /home/${NB_USER}/Karabo-Pipeline

# Ensure the karabo env auto-activates for the default user in interactive and non-interactive shells
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate karabo" >> /home/${NB_USER}/.bashrc && \
    mkdir -p /opt/etc && \
    echo "source /opt/conda/etc/profile.d/conda.sh && conda activate karabo" > /opt/etc/conda_init_script && \
    chmod 644 /opt/etc/conda_init_script && \
    chown -R ${NB_UID}:${NB_GID} /opt/etc /home/${NB_USER}

ENV BASH_ENV=/opt/etc/conda_init_script

# Switch to jovyan for user-scoped installs
USER ${NB_UID}

# Download test data before running tests
RUN cd /home/${NB_USER}/Karabo-Pipeline && python download_test_data.py

# Register Jupyter kernel for jovyan
RUN python -m ipykernel install --user --name=karabo --display-name="Karabo (Python 3.9)"

# Run tests to ensure everything works (this will use the downloaded test data)
RUN cd /home/${NB_USER}/Karabo-Pipeline && python -m pytest -x --tb=short

# Set working directory
WORKDIR "/home/${NB_USER}"
