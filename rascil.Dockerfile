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

ARG NUMPY_VERSION=1.23.5
ARG PYTHON_VERSION=3.10

# install base dependencies before adding extra spack overlays, this avoids extra build time
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
        'py-numpy@'$NUMPY_VERSION \
        'py-pip@:25.2' \
        'python@'$PYTHON_VERSION \
        # DO NOT ADD ANYTHING HERE
    && \
    spack concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast && \
    spack env view regenerate

# Version pins aligned with sp5505
ARG ASTROPLAN_VERSION=0.8
ARG ASTROPY_HEALPIX_VERSION=1.0.0
ARG ASTROPY_VERSION=5.1
ARG BDSF_VERSION=1.12.0
ARG CASACORE_VERSION=3.5.0
ARG DASK_VERSION=2022.10.2
ARG DISTRIBUTED_VERSION=2022.10.2
ARG DUCC_VERSION=0.27
ARG H5PY_VERSION=3.7
ARG HDF5_VERSION=1.12.3
ARG HEALPY_VERSION=1.16.2
ARG MATPLOTLIB_VERSION=3.6
ARG PANDAS_VERSION=1.5.3
ARG PEYERFA_VERSION=2.0.0
ARG PHOTUTILS_VERSION=1.11.0
ARG REPROJECT_VERSION=0.9.1
ARG SCIPY_VERSION=1.9.3
ARG SDP_DATAMODELS_VERSION=0.1.3
ARG SDP_FUNC_VERSION=0.1.5
ARG XARRAY_VERSION=2022.12.0
ARG RASCIL_VERSION=1.0.0

# Add SKA SDP Spack repo and overlay
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/ska-sdp-spack
COPY spack-overlay /opt/karabo-spack
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/karabo-spack

RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    spack env activate /opt/spack_env; \
    spack add \
        'boost+python+numpy' \
        'casacore@'$CASACORE_VERSION'+python' \
        'cfitsio' \
        'fftw~mpi~openmp' \
        'hdf5@'$HDF5_VERSION'+hl~mpi' \
        'openblas@:0.3.27' \
        'py-astroplan@'$ASTROPLAN_VERSION \
        'py-astropy-healpix@'$ASTROPY_HEALPIX_VERSION \
        'py-astropy@'$ASTROPY_VERSION \
        'py-bdsf@'$BDSF_VERSION \
        'py-casacore@'$CASACORE_VERSION \
        'py-dask@'$DASK_VERSION \
        'py-distributed@'$DISTRIBUTED_VERSION \
        'py-ducc@'$DUCC_VERSION \
        'py-h5py@'$H5PY_VERSION \
        'py-healpy@'$HEALPY_VERSION \
        'py-matplotlib@'$MATPLOTLIB_VERSION \
        'py-numpy@'$NUMPY_VERSION \
        'py-pandas@'$PANDAS_VERSION \
        'py-photutils@'$PHOTUTILS_VERSION \
        'py-pyerfa@'$PEYERFA_VERSION \
        'py-reproject@'$REPROJECT_VERSION \
        'py-scipy@'$SCIPY_VERSION \
        'py-seqfile@0.2.0' \
        'py-ska-sdp-datamodels@'$SDP_DATAMODELS_VERSION \
        'py-ska-sdp-func-python@'$SDP_FUNC_VERSION \
        'py-ska-sdp-func@0.1.0' \
        'py-tabulate' \
        'py-xarray@'$XARRAY_VERSION \
    && \
    spack concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast && \
    spack env view regenerate && \
    spack test run 'py-astroplan' && \
    spack test run 'py-astropy-healpix' && \
    spack test run 'py-astropy' && \
    spack test run 'py-bdsf' && \
    spack test run 'py-casacore' && \
    spack test run 'py-ducc' && \
    spack test run 'py-h5py' && \
    spack test run 'py-rascil' && \
    spack test run 'py-numpy' && \
    spack test run 'py-pandas' && \
    spack test run 'py-photutils' && \
    spack test run 'py-pyerfa' && \
    spack test run 'py-reproject' && \
    spack test run 'py-scipy' && \
    spack test run 'py-seqfile' && \
    spack test run 'py-ska-sdp-datamodels' && \
    spack test run 'py-ska-sdp-func-python' && \
    spack test run 'py-ska-sdp-func' && \
    spack test run 'py-xarray'

# Install dask-memusage without using buildcache/mirrors and then RASCIL
RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    spack env activate /opt/spack_env; \
    (spack mirror remove mycache || true); \
    spack add 'py-dask-memusage@1.1' && \
    spack concretize --force && \
    spack install --no-cache --no-check-signature --no-checksum; \
    spack add 'py-rascil@'$RASCIL_VERSION && \
    spack concretize --force && \
    spack install --no-check-signature --no-checksum 'py-rascil@'$RASCIL_VERSION && \
    spack env view regenerate && \
    spack test run 'py-rascil'

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

# Minimal validation of dependencies
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python - <<'PY'
import importlib
pkgs = [
    ('astropy','5.1'),
    ('bdsf','1.10'),
    ('casacore','3.5'),
    ('click','0.0'),
    ('cloudpickle','0.0'),
    ('dask','2022.0'),
    ('distributed','2022.0'),
    ('ducc0','0.27'),
    ('fsspec','0.0'),
    ('h5py','3.7'),
    ('locket','0.0'),
    ('matplotlib','3.6'),
    ('msgpack','0.0'),
    ('natsort','0.0'),
    ('numpy','1.23'),
    ('pandas','1.5'),
    ('partd','0.0'),
    ('photutils','1.11'),
    ('reproject','0.9'),
    ('astroplan','0.8'),
    ('scipy','1.9'),
    ('seqfile','0.2'),
    ('ska_sdp_datamodels','0.1'),
    ('ska_sdp_func_python','0.1'),
    ('ska_sdp_func','0.0'),
    ('sortedcontainers','0.0'),
    ('tabulate','0.9'),
    ('tblib','0.0'),
    ('toolz','0.0'),
    ('xarray','2022.12'),
    ('zict','0.0'),
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
        # ensure astropy <5.2
        if name == 'astropy':
            if ver_tup >= (5,2):
                print(f'FAIL {name}: {ver} >= 5.2')
                exit(1)
        target_tup=tuple(map(int,target.split('.')))
        if ver_tup < target_tup:
            print(f'FAIL {name}: {ver} < {target}')
            exit(1)
        print(f'OK {name} installed={ver}, target={target}')
    else:
        print(f'OK {name} installed=???')
print('ALL_DEPS_OK')
PY

# Default user back to jovyan
USER ${NB_UID}
WORKDIR /home/${NB_USER}

CMD ["bash"]


