FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2

# RASCIL deps-only image (no RASCIL itself). Versions aligned with sp5505.
USER root
SHELL ["/bin/bash", "-lc"]

# Remove conda to avoid library conflicts
RUN rm -f /usr/local/bin/before-notebook.d/10activate-conda-env.sh || true; \
    rm -rf /opt/conda || true

# Essential system dependencies
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
        patchelf \
        pkg-config \
        wget \
        zstd \
    ;

# Install Rust before any Spack setup, because Spack rust is unbelievably slow.
ARG RUST_VERSION=1.81.0
ENV CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain $RUST_VERSION --no-modify-path && \
    ln -sf /opt/cargo/bin/* /usr/local/bin/ && \
    rustc --version | grep -Fq "$RUST_VERSION"

ENV SPACK_ROOT=/opt/spack \
    SPACK_DISABLE_LOCAL_CONFIG=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Spack v0.23 and detect compilers
RUN git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ${SPACK_ROOT} && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack compiler find && \
    spack external find rust && \
    spack external find git && \
    spack external find pkgconf

ARG NUMPY_VERSION=1.26.4
ARG PYTHON_VERSION=3.10
ARG DUCC_VERSION=0.27.0

# Add SKA SDP Spack repo and overlay
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/ska-sdp-spack
COPY spack-overlay /opt/karabo-spack
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/karabo-spack

RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    mkdir -p /opt/{software,view,buildcache,spack-source-cache,spack-misc-cache}; \
    spack env create --dir /opt/spack_env; \
    spack env activate /opt/spack_env; \
    spack config add "config:install_tree:root:/opt/software"; \
    spack config add "concretizer:unify:true"; \
    spack config add "concretizer:reuse:true"; \
    spack config add "view:/opt/view"; \
    spack config add "config:source_cache:/opt/spack-source-cache"; \
    spack config add "config:misc_cache:/opt/spack-misc-cache"; \
    spack mirror add --autopush --unsigned mycache file:///opt/buildcache; \
    spack buildcache keys --install --trust || true; \
    spack add \
        'python@'$PYTHON_VERSION \
        'py-pip@:25.2' \
        'py-numpy@'$NUMPY_VERSION \
        'py-ducc@'$DUCC_VERSION \
        "py-setuptools" \
        "py-wheel" \
        "py-build" \
        "py-setuptools-scm" \
        "cmake@3.18:" \
    && \
    spack --verbose concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast --reuse || ( \
        for f in /tmp/root/spack-stage/spack-stage-py-ducc-0.27.0-*/spack-build-out.txt; do \
            echo "=== $f ==="; \
            cat $f; \
        done; \
        exit 1; \
    ) && \
    spack env view regenerate && \
    spack test run 'py-setuptools' && \
    spack test run 'py-wheel' && \
    spack test run 'py-build' && \
    spack test run 'py-setuptools-scm' && \
    spack test run 'cmake' && \
    spack test run 'py-ducc'

