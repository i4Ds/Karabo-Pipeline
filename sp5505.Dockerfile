# syntax=docker/dockerfile:1.6
FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2

# Spack-only environment (no conda usage). Ensure Spack Python is default for NB_UID shells
USER root
SHELL ["/bin/bash", "-lc"]

# System dependencies for building scientific stack
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
    wget ca-certificates curl git file build-essential gfortran cmake pkg-config zstd patchelf \
    libcfitsio-dev libfftw3-dev libhdf5-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV SPACK_ROOT=/opt/spack \
    SPACK_DISABLE_LOCAL_CONFIG=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Spack v0.23 and detect compilers
RUN git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ${SPACK_ROOT} && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack compiler find

# Add SKA SDP Spack repo
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/ska-sdp-spack

# Create Spack environment, enable view at /opt/view, add packages, install
# spack env deactivate || true; \
# spack env rm -y /opt/spack_env || true; \
# spack mirror rm -y mycache || true; \
RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    mkdir -p /opt/{software,view,buildcache,spack-source-cache,spack-misc-cache}; \
    spack env create --dir /opt/spack_env; \
    spack env activate /opt/spack_env; \
    spack config add "config:install_tree:root:/opt/software"; \
    spack config add "concretizer:unify:when_possible"; \
    spack config add "view:/opt/view"; \
    spack config add "config:source_cache:/opt/spack-source-cache"; \
    spack config add "config:misc_cache:/opt/spack-misc-cache"; \
    spack mirror add --autopush --unsigned mycache file:///opt/buildcache; \
    spack buildcache keys --install --trust || true; \
    spack add \
      "python@3.10" \
      "openblas@0.3.27" \
      "mpich" \
      "boost+python+numpy" \
      "py-numpy@=1.23.5" \
      "py-scipy@=1.9.3" \
      "py-pandas@=1.5.3" \
      "py-matplotlib@=3.6.3" \
      "py-h5py@=3.7.0" \
      "py-pip@=22.1.2" \
      "py-setuptools@=59.4.0" \
      "py-wheel@=0.37.1" \
      "py-pybind11@=2.13.5" \
      "py-pytest"; \
    spack concretize --force && \
    spack install --no-check-signature --fail-fast -j $(nproc) && \
    spack env view regenerate

# Make Spack view default in system paths and shells
RUN printf "/opt/view/lib\n/opt/view/lib64\n" > /etc/ld.so.conf.d/spack-view.conf && ldconfig && \
    echo ". ${SPACK_ROOT}/share/spack/setup-env.sh 2>/dev/null || true" > /etc/profile.d/spack.sh && \
    echo "spack env activate /opt/spack_env 2>/dev/null || true" >> /etc/profile.d/spack.sh && \
    mkdir -p /opt/etc && \
    echo ". /etc/profile.d/spack.sh" > /opt/etc/spack_env && \
    chmod 644 /opt/etc/spack_env

# Prefer Spack view in PATH, and ensure batch shells source spack env
ENV PATH="/opt/view/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64:${LD_LIBRARY_PATH}" \
    BASH_ENV=/opt/etc/spack_env \
    PYTHONNOUSERSITE=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Copy repository for editable install and testing
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100
COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/Karabo-Pipeline
RUN fix-permissions /home/${NB_USER}/Karabo-Pipeline

# Install ipykernel, pytest, and karabo-pipeline into Spack Python (as root)
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh; spack env activate /opt/spack_env; \
    python -m pip install --no-cache-dir --no-build-isolation ipykernel pytest -e /home/jovyan/Karabo-Pipeline

# Register kernel for jovyan user using the Spack Python
USER ${NB_UID}
RUN python -m ipykernel install --user --name=karabo --display-name="Karabo (Spack Py3.9)"

# Run tests during build to validate environment
ARG SKIP_TESTS=0
ENV SKIP_TESTS=${SKIP_TESTS}
RUN if [ "${SKIP_TESTS:-0}" = "1" ]; then exit 0; fi; pytest -q -x --tb=short /home/${NB_USER}/Karabo-Pipeline

WORKDIR "/home/${NB_USER}"
