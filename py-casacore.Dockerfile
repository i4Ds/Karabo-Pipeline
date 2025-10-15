FROM ubuntu:24.04

# docker build -f py-casacore.Dockerfile -t py-casacore . 2>&1 | tee py-casacore.log

USER root
SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    SPACK_ROOT=/opt/spack \
    SPACK_DISABLE_LOCAL_CONFIG=1

# Minimal system deps for building with Spack
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        file \
        gfortran \
        git \
        python3 \
        python3-venv \
        patchelf \
        pkg-config \
        wget \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Spack v0.23 and detect compilers
RUN git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ${SPACK_ROOT} && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack external find python && \
    spack compiler find && \
    spack external find git && \
    spack external find pkgconf

# Add SKA SDP Spack repo for py-casacore
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack repo add /opt/ska-sdp-spack

# Create a view, install only Python and py-casacore
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    mkdir -p /opt/view; \
    spack env create --dir /opt/spack_env; \
    spack env activate /opt/spack_env; \
    spack config add "view:/opt/view"; \
    spack add python@3.10 py-casacore; \
    spack concretize --force; \
    spack install --no-check-signature --no-checksum --fail-fast --reuse; \
    spack env view regenerate

# test py-casacore
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    spack test run py-casacore

# Ensure Spack view is used by default
ENV PATH=/opt/view/bin:$PATH

# Quick import test during build to verify installation
RUN /opt/view/bin/python3 - <<'PY'
import sys
try:
    import casacore
    import casacore.tables as tb
    print("py-casacore OK")
except Exception as e:
    print("py-casacore import failed:", e)
    sys.exit(1)
PY

# Default command prints installed module path
CMD ["/opt/view/bin/python3", "-c", "import casacore, sys; print(casacore.__file__) "]


