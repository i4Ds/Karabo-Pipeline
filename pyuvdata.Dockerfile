# syntax=docker/dockerfile:1.7-labs
FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2

USER root
SHELL ["/bin/bash", "-lc"]

# Remove conda to avoid library conflicts
RUN rm -f /usr/local/bin/before-notebook.d/10activate-conda-env.sh || true; \
    rm -rf /opt/conda || true

# Essential system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        file \
        gfortran \
        git \
        libtool \
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
    spack external find pkgconf && \
    spack external find autoconf && \
    spack external find automake && \
    spack external find libtool

# Add SKA SDP Spack repo and overlay
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/ska-sdp-spack
COPY spack-overlay /opt/karabo-spack
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/karabo-spack

ARG ASTROPY_VERSION=5.1.1
ARG ASTROPY_HEALPIX_VERSION=1.1.2
ARG MATPLOTLIB_VERSION=3.9.2
# conda uses 3.10.5 but max available is 3.9.2
ARG NUMPY_VERSION=1.23.5
# astropy 5.1.1 requires numpy<1.24; healpy 1.16.6 builds with 1.23.x
ARG CFITSIO_VERSION=4.3.1
# conda installs 4.3.1
# ARG PYERFA_VERSION=2.0.1.5
# just let astropy install its own erfa
# astropy 5.1.1 requires pyerfa <2.0.1.0
ARG PYTHON_VERSION=3.10
# conda installs 3.10.18, but only up to 3.10.14 is available in spack
ARG SCIPY_VERSION=1.9.3
# SciPy 1.9.3 pairs with NumPy 1.23.5 for older C-API compatibility
# conda uses scipy 1.13.1 but this requires cupy and torch
ARG HEALPY_VERSION=1.16.6
# conda installs 1.16.6
ARG PYUVDATA_VERSION=2.4.2
# conda installs 2.4.2

# install base dependencies before adding extra spack overlays, this avoids extra build time
# Create Spack environment and install deps (no RASCIL)
RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    mkdir -p /opt/{software,view,buildcache,spack-source-cache,spack-misc-cache}; \
    spack env create --dir /opt/spack_env; \
    spack env activate /opt/spack_env; \
    spack clean && \
    spack config add "config:install_tree:root:/opt/software"; \
    spack config add "packages:py-astropy:version:[${ASTROPY_VERSION}]"; \
    spack config add "packages:py-astropy-healpix:version:[${ASTROPY_HEALPIX_VERSION}]"; \
    spack config add "packages:py-matplotlib:version:[${MATPLOTLIB_VERSION}]"; \
    spack config add "packages:py-numpy:version:[${NUMPY_VERSION}]"; \
    # spack config add "packages:py-pyerfa:version:[${PYERFA_VERSION}]"; \
    spack config add "packages:py-scipy:version:[${SCIPY_VERSION}]"; \
    spack config add "packages:py-healpy:version:[${HEALPY_VERSION}]"; \
    spack config add "concretizer:unify:true"; \
    spack config add "view:/opt/view"; \
    spack config add "config:source_cache:/opt/spack-source-cache"; \
    spack config add "config:misc_cache:/opt/spack-misc-cache"; \
    spack mirror add --autopush --unsigned mycache file:///opt/buildcache; \
    spack buildcache keys --install --trust || true; \
    spack add \
        'python@'$PYTHON_VERSION \
        'py-numpy@'$NUMPY_VERSION \
        'cfitsio@'$CFITSIO_VERSION \
        'py-scipy@'$SCIPY_VERSION \
        # 'py-pyerfa@'$PYERFA_VERSION \
        'py-matplotlib@'$MATPLOTLIB_VERSION \
        'py-astropy@'$ASTROPY_VERSION \
        'py-healpy@'$HEALPY_VERSION'+internal-healpix' \
        'py-astropy-healpix@'$ASTROPY_HEALPIX_VERSION \
        'py-pyuvdata@'$PYUVDATA_VERSION'+casa' \
    && \
    spack concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast && \
    spack env view regenerate

# Make Spack view default in PATH and shells
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

RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate -p /opt/spack_env && \
    spack test run 'py-numpy' && \
    spack test run 'py-scipy' && \
    spack test run 'py-pyerfa' && \
    # spack test run 'py-astropy'
    # The Spack-driven py-astropy test step is still failing because the harness tries to import the Astropy ASDF test packages, which in turn import plain pytest, and the sandbox it runs in doesn't have pytest on PYTHONPATH. Even though py-pytest was installed as a root spec, the test runner launches /opt/software/.../python3 in a clean staging environment and only populates it with the dependencies declared in the package. Astropy's packaging treats most of these test modules as optional; they aren't pulled in automatically, so the runner reports ModuleNotFoundError: No module named 'pytest'.
    spack test run 'py-pyuvdata'

# Minimal validation of dependencies
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python - <<'PY'
import importlib
pkgs = [
    ('astropy','5.1'),
    ('astropy_healpix', '1.1'),
    ('matplotlib','3.6'),
    ('numpy','1.23'),
    ('erfa','2.0'),
    ('scipy','1.9'),
    ('pyuvdata','2.4.2'),
]
for name, target in pkgs:
    try:
        m = importlib.import_module(name)
    except Exception as e:
        print(f'IMPORT_FAIL {name}: {e}')
        exit(1)
    try:
        ver = getattr(m, '__version__')
    except Exception as e:
        ver = None
    if ver:
        ver_tup=tuple(map(int,ver.split('.')))
        target_tup=tuple(map(int,target.split('.')))
        if ver_tup < target_tup:
            print(f'FAIL {name}: {ver} < {target}')
            exit(1)
        print(f'OK {name} installed={ver}, target={target}')
    else:
        print(f'OK {name} installed=???')
print('ALL_DEPS_OK')
PY

# Default user back to jovyan for safety
USER ${NB_UID}
WORKDIR /home/${NB_USER}

# Run tests by default; all output is tee'd inside script
CMD ["bash"]
