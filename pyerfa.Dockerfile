FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2

USER root
SHELL ["/bin/bash", "-lc"]

# Remove conda to avoid library conflicts
RUN rm -f /usr/local/bin/before-notebook.d/10activate-conda-env.sh || true; \
    rm -rf /opt/conda || true

# Essential build dependencies
# These are found by spack external find, and later garbage collected by spack.
# Do not include runtime dependencies here.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
        autoconf \
        automake \
        bison \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        diffutils \
        file \
        findutils \
        gfortran \
        git \
        libcurl4-openssl-dev \
        libtool \
        m4 \
        meson \
        patchelf \
        perl \
        pkg-config \
        wget \
        zstd \
    ; # not required because of buildcache: rm -rf /var/lib/apt/lists/*

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
RUN git clone --depth=1 --single-branch --branch=releases/v0.23 https://github.com/spack/spack.git ${SPACK_ROOT} && \
    rm -rf ${SPACK_ROOT}/.git && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack compiler find && \
    spack external find \
        autoconf \
        automake \
        bison \
        curl \
        diffutils \
        findutils \
        git \
        libtool \
        m4 \
        meson \
        perl \
        pkgconf \
        rust

# Add SKA SDP Spack repo and overlay
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/ska-sdp-spack
COPY spack-overlay /opt/karabo-spack
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/karabo-spack

ARG NUMPY_VERSION=1.23.5
# astropy 5.1.1 requires numpy<1.24; healpy 1.16.6 builds with 1.23.x
ARG PYTHON_VERSION=3.10
# conda installs 3.10.18, but only up to 3.10.14 is available in spack
ARG PYERFA_VERSION=2.0.1.5

# Create Spack environment and install py-healpy with internal HEALPix C++
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
        'python@'$PYTHON_VERSION \
        'py-numpy@'$NUMPY_VERSION \
        'py-pyerfa@'$PYERFA_VERSION \
    && \
    spack concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast --reuse && \
    spack gc -y && \
    spack env view regenerate

# Make Spack view default in system paths and shells
RUN printf "/opt/view/lib\n/opt/view/lib64\n" > /etc/ld.so.conf.d/spack-view.conf && ldconfig && \
    echo ". ${SPACK_ROOT}/share/spack/setup-env.sh 2>/dev/null || true" > /etc/profile.d/spack.sh && \
    echo "spack env activate -p /opt/spack_env 2>/dev/null || true" >> /etc/profile.d/spack.sh && \
    mkdir -p /opt/etc && \
    echo ". /etc/profile.d/spack.sh" > /opt/etc/spack_env && \
    chmod 644 /opt/etc/spack_env

ENV PATH="/opt/view/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64" \
    BASH_ENV=/opt/etc/spack_env \
    PYTHONNOUSERSITE=1 \
    CMAKE_PREFIX_PATH="/opt/view" \
    PKG_CONFIG_PATH="/opt/view/lib/pkgconfig:/opt/view/lib64/pkgconfig"

# Ensure start-notebook uses Spack jupyter first in PATH
RUN mkdir -p /usr/local/bin/before-notebook.d && \
    printf '#!/usr/bin/env bash\nPATH="/opt/view/bin:${HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nexport PATH\nLD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64"\nexport LD_LIBRARY_PATH\n' > /usr/local/bin/before-notebook.d/00-prefer-spack.sh && \
    chmod +x /usr/local/bin/before-notebook.d/00-prefer-spack.sh && \
    # Remove conda and activation hook; we run Jupyter inside Spack Python
    rm -f /opt/conda /usr/local/bin/before-notebook.d/10activate-conda-env.sh || true

# Replace spack test run ... healpy
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    /opt/view/bin/python - <<'PY'
import sys, erfa, astropy, healpy
print(sys.executable)
# Minimal assertions to mirror what failed under venv:
from erfa import DAYSEC, ELG  # must exist
import importlib; importlib.import_module('healpy._pixelfunc')
print('OK: erfa + healpy imports and constants present')
PY

RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -c 'import erfa; print("\n".join(f"{attr}: {getattr(erfa,attr)}" for attr in dir(erfa)))'

RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate -p /opt/spack_env && \
    spack test run 'py-numpy' && \
    spack test run 'py-pyerfa' && \
# Minimal import test for healpy
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python - <<'PY'
import sys, os, pprint
print('sys.executable:', sys.executable)
print('PYTHONPATH:', os.environ.get('PYTHONPATH'))
print('sys.path=')
pprint.pprint(sys.path)
import importlib

pkgs = [
    ('numpy','1.23'),
    ('erfa','2.0'),
]

def parse_version(v):
    parts = []
    for p in v.split('.'):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts)

for name, target in pkgs:
    m = importlib.import_module(name)
    ver = getattr(m, '__version__', None)
    if ver:
        target_tup = parse_version(target)
        ver_tup = parse_version(ver)
        if ver_tup and target_tup and ver_tup < target_tup:
            print(f'FAIL {name}: {ver} < {target}')
            raise SystemExit(1)
        print(f'OK {name} installed={ver}, target={target}')
    else:
        print(f'OK {name} installed=???')
print('ALL_DEPS_OK')
PY
