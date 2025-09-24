# syntax=docker/dockerfile:1.6
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
        libcurl4-openssl-dev \
        patchelf \
        pkg-config \
        wget \
        zstd \
    ;

ENV SPACK_ROOT=/opt/spack \
    SPACK_DISABLE_LOCAL_CONFIG=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Spack v0.23 and detect compilers
RUN git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ${SPACK_ROOT} && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack compiler find

# Add SKA SDP Spack repo and overlay
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/ska-sdp-spack
COPY spack-overlay /opt/karabo-spack
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/karabo-spack

# Version pins aligned with sp5505
ARG NUMPY_VERSION=1.23.5
ARG PANDAS_VERSION=1.5.3
ARG XARRAY_VERSION=2022.12.0
ARG H5PY_VERSION=3.7
ARG HDF5_VERSION=1.12.3
ARG SCIPY_VERSION=1.9.3
ARG MATPLOTLIB_VERSION=3.6
ARG ASTROPY_VERSION=5.3
ARG CASACORE_VERSION=3.5.0
ARG HEALPY_VERSION=1.16.2
ARG PEYERFA_VERSION=2.0.0
ARG ASTROPY_HEALPIX_VERSION=1.0.0
ARG REPROJECT_VERSION=0.9.1
ARG PHOTUTILS_VERSION=1.11.0
ARG DUCC_VERSION=0.27

# Create Spack environment and install deps (no RASCIL)
RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    mkdir -p /opt/{software,view,buildcache,spack-source-cache,spack-misc-cache}; \
    spack env create --dir /opt/spack_env; \
    spack env activate /opt/spack_env; \
    spack config add "config:install_tree:root:/opt/software"; \
    spack config add "concretizer:unify:true"; \
    spack config add "view:/opt/view"; \
    spack config add "config:source_cache:/opt/spack-source-cache"; \
    spack config add "config:misc_cache:/opt/spack-misc-cache"; \
    spack mirror add --autopush --unsigned mycache file:///opt/buildcache; \
    spack buildcache keys --install --trust || true; \
    spack add \
        'boost+python+numpy' \
        'cfitsio' \
        'curl' \
        'fftw~mpi~openmp' \
        'hdf5@'$HDF5_VERSION'+hl~mpi' \
        'openblas@:0.3.27' \
        'casacore@'$CASACORE_VERSION'+python' \
        'py-astropy-healpix@'$ASTROPY_HEALPIX_VERSION \
        'py-reproject@'$REPROJECT_VERSION \
        'py-astropy@'$ASTROPY_VERSION \
        'py-photutils@'$PHOTUTILS_VERSION \
        'py-ducc@'$DUCC_VERSION \
        'py-casacore@'$CASACORE_VERSION \
        'py-dask@2022.10.2' \
        'py-distributed@2022.10.2' \
        'py-h5py@'$H5PY_VERSION \
        'py-healpy@'$HEALPY_VERSION \
        'py-ipykernel' \
        'py-matplotlib@'$MATPLOTLIB_VERSION \
        'py-mpi4py' \
        'py-nbconvert' \
        'py-numpy@'$NUMPY_VERSION \
        'py-pandas@'$PANDAS_VERSION \
        'py-pip@:25.2' \
        'py-pyerfa@'$PEYERFA_VERSION \
        'py-pytest' \
        'py-pyyaml' \
        'py-requests' \
        'py-scipy@'$SCIPY_VERSION \
        'py-tabulate' \
        'py-tornado@6.1' \
        'py-xarray@'$XARRAY_VERSION \
        'python@3.10' \
        'zlib' \
    && \
    spack concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast python@3.10 py-pip py-numpy && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast && \
    spack env view regenerate

RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    spack env activate /opt/spack_env; \
    spack test run 'py-astropy-healpix@'$ASTROPY_HEALPIX_VERSION && \
    spack test run 'py-astropy@'$ASTROPY_VERSION && \
    spack test run 'py-casacore@'$CASACORE_VERSION \ && \
    spack test run 'py-h5py@'$H5PY_VERSION && \
    spack test run 'py-numpy@'$NUMPY_VERSION && \
    spack test run 'py-pandas@'$PANDAS_VERSION && \
    spack test run 'py-photutils@'$PHOTUTILS_VERSION && \
    spack test run 'py-pyerfa@'$PEYERFA_VERSION && \
    spack test run 'py-reproject@'$REPROJECT_VERSION && \
    spack test run 'py-scipy@'$SCIPY_VERSION && \
    spack test run 'py-xarray@'$XARRAY_VERSION && \
    spack test run 'py-ducc@'$DUCC_VERSION
# others:
# spack test run py-ska_sdp_datamodels && \
# spack test run py-ska_sdp_func_python && \

# Make Spack view default in PATH and shells
RUN printf "/opt/view/lib\n/opt/view/lib64\n" > /etc/ld.so.conf.d/spack-view.conf && ldconfig && \
    echo ". ${SPACK_ROOT}/share/spack/setup-env.sh 2>/dev/null || true" > /etc/profile.d/spack.sh && \
    echo "spack env activate /opt/spack_env 2>/dev/null || true" >> /etc/profile.d/spack.sh && \
    mkdir -p /opt/etc && \
    echo ". /etc/profile.d/spack.sh" > /opt/etc/spack_env && \
    chmod 644 /opt/etc/spack_env

ENV PATH="/opt/view/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64:${LD_LIBRARY_PATH}" \
    BASH_ENV=/opt/etc/spack_env \
    PYTHONNOUSERSITE=1 \
    CMAKE_PREFIX_PATH="/opt/view:${CMAKE_PREFIX_PATH}" \
    PKG_CONFIG_PATH="/opt/view/lib/pkgconfig:/opt/view/lib64/pkgconfig:${PKG_CONFIG_PATH}"

# Pip-only pins required by RASCIL that are not available/pinned suitably in Spack
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    /opt/view/bin/python -m pip install --no-build-isolation --no-deps -U 'pip<25.3' setuptools setuptools-scm wheel build 'extension-helpers>=1.0,<2' && \
    # ducc0 now installed via Spack
    # ska-sdp-datamodels with core pins
    # /opt/view/bin/python -m pip install --no-build-isolation \
    #     --index-url=https://artefact.skao.int/repository/pypi-all/simple \
    #     --extra-index-url=https://pypi.org/simple \
    #     'h5py=='$H5PY_VERSION \
    #     'numexpr==2.10.2' \
    #     'xarray=='$XARRAY_VERSION && \
    /opt/view/bin/python -m pip install --no-build-isolation \
        --index-url=https://artefact.skao.int/repository/pypi-all/simple \
        --extra-index-url=https://pypi.org/simple \
        'ska-sdp-datamodels==0.1.3' && \
    # ska-sdp-func-python and ska-sdp-func
    /opt/view/bin/python -m pip install --no-build-isolation --no-deps \
        --index-url=https://artefact.skao.int/repository/pypi-all/simple \
        --extra-index-url=https://pypi.org/simple \
        'ska-sdp-func-python==0.1.5' \
        'git+https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git@08eb17cf9f4d63320dd0618032ddabc6760188c9'

# Minimal validation of dependencies
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python - <<'PY'
import importlib
pkgs = [
    ('numpy','1.23'),
    ('scipy','1.9'),
    ('pandas','1.5'),
    ('xarray','2022.12'),
    ('astropy','5.3'),
    ('matplotlib','3.6'),
    ('h5py','3.7'),
    ('casacore','3.5'),
    ('reproject','0.9'),
    ('photutils','1.11'),
    ('ducc0','0.27'),
    ('ska_sdp_datamodels','0.1'),
    ('ska_sdp_func_python','0.1'),
]
for name, target in pkgs:
    try:
        m = importlib.import_module(name)
    except Exception as e:
        print(f'IMPORT_FAIL {name}: {e}')
        raise SystemExit(1)
    ver = getattr(m, '__version__', '0.0')
    print(f'OK {name} {ver}')
print('ALL_DEPS_OK')
PY

# Default user back to jovyan
USER ${NB_UID}
WORKDIR /home/${NB_USER}

CMD ["bash"]


