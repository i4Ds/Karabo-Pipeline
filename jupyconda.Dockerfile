FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2

# Install Karabo via conda
USER root
# Create new environment with Python 3.9
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda install -y -n base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda create -y -n karabo python=3.9
# install karabo-pipeline in the karabo environment
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda install -y -n karabo -c i4ds -c conda-forge -c "nvidia/label/cuda-11.7.1" karabo-pipeline && \
    conda run -n karabo pip install ipykernel && \
    conda run -n karabo python -m ipykernel install --user --name=karabo --display-name="Karabo (Python 3.9)" && \
    echo "/opt/conda/envs/karabo/lib" > /etc/ld.so.conf.d/conda.conf && \
    ldconfig && \
    fix-permissions /home/$NB_USER

# clone karabo-pipeline from github and test that OSKAR works
RUN git clone https://github.com/i4ds/karabo-pipeline.git /karabo && \
    fix-permissions /karabo

# Switch back to jovyan user
USER ${NB_UID}

# Set working directory
WORKDIR "${HOME}"

