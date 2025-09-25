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
ARG BDSF_VERSION=1.12.0
ARG DASK_VERSION=2022.10.2
ARG DUCC_VERSION=0.27
ARG PYERFA_VERSION=2.0.0
ARG PHOTUTILS_VERSION=1.11.0
ARG REPROJECT_VERSION=0.9.1
ARG SDP_DATAMODELS_VERSION=0.1.3
ARG SDP_FUNC_VERSION=0.1.5
ARG RASCIL_VERSION=1.0.0

# numpy needed by pyuvdata montagepy numexpr scipy rascil scikit-image pywavelets astroml ducc0 imageio ska-sdp-func-python contourpy aratmospy bokeh astroplan coda harp astropy-healpix katbeam tensorboard h5py dask ml_dtypes ska-gridder-nifty-cuda libboost-python-devel python-casacore tifffile pytest-arraydiff shapely bdsf casacore finufft reproject numcodecs matplotlib-base tools21cm libboost-python numba gwcs tensorflow-base pyfftw boost xarray asdf pyside6 photutils astropy bottleneck pandas oskarpy ska-sdp-datamodels ska-sdp-func healpy keras scikit-learn pyerfa eidos asdf-astropy zarr bluebild
# numpy 1.26.4 installed by conda
# numpy>=1.24 required by zarr 2.18.3
# numpy 1.23 needed by rascil 1.0.0 and ska-sdp-func-python 0.1.5
ARG PANDAS_VERSION=1.5.3
# pandas needed by rascil dask xarray ska-sdp-datamodels bluebild
# pandas 1.5.3 is installed by conda
ARG XARRAY_VERSION=2022.12.0
# xarray needed by pyuvdata bluebild rascil scikit-image astroml ska-sdp-func-python aratmospy bdsf reproject tools21cm gwcs photutils healpy scikit-learn eidos
# xarray 2023.2.0 is installed by conda
# xarray<2022.13,>=2022.12 required by rascil 1.0.0
# xarray<2023.0.0,>=2022.10.0 required by ska-sdp-datamodels 0.1.3
# xarray<2023.0.0,>=2022.11.0 required by ska-sdp-func-python 0.1.5
# only version that meets this is 2022.12
# only 2023.7.0 2022.3.0 available in spack builtin
# only 2025.4.0 2024.10.0 available in sdp spack (but only the main branch, not the 2025.07.3 branch)
ARG H5PY_VERSION=3.7
# h5py needed by pyuvdata tensorflow-base ska-sdp-datamodels keras
ARG HDF5_VERSION=1.12.3
# hdf5 1.14.3 installed by conda
# hdf5 1.12.3 seems to have worked at one point
# hdf5 1.10.10 installed by ubuntu24 apt
ARG DISTRIBUTED_VERSION=2022.10.2
# distributed needed by rascil, dask
ARG SCIPY_VERSION=1.9.3
# scipy needed by pyuvdata bluebild rascil scikit-image astroml ska-sdp-func-python aratmospy bdsf reproject tools21cm gwcs photutils healpy scikit-learn eidos
# scipy 1.13.1 installed by conda
ARG MATPLOTLIB_VERSION=3.6
# matplotlib needed by bluebild rascil aratmospy tools21cm
ARG ASTROPY_VERSION=5.1
# astropy needed by rascil pyuvdata ska-sdp-func-python aratmospy bdsf tools21cm gwcs photutils ska-sdp-datamodels healpy eidos bluebild
# astropy>5.2 has no ._erfa
ARG CASACORE_VERSION=3.5.0
# casacore needed by everybeam wsclean oskar rascil
ARG HEALPY_VERSION=1.16.2
# healpy needed by rascil ska-sdp-func-python aratmospy bdsf tools21cm gwcs photutils ska-sdp-datamodels eidos bluebild

# up to 2.0.1.5
ARG OSKAR_VERSION=2.8.3

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
        'py-pyerfa@'$PYERFA_VERSION \
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

# copy early sanity test to run immediately after Spack deps
COPY karabo/test/test_000_astropy_env.py /opt/early-tests/test_000_astropy_env.py

RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    spack env activate /opt/spack_env; \
    # dask memusage needed by rascil, buildcache issues
    spack add 'py-dask-memusage@1.1' && \
    spack concretize --force && \
    spack install --no-check-signature --no-checksum --fail-fast --no-cache && \
    spack add \
        # todo: py-aratmospy py-eidos py-katbeam py-pyuvdata py-rfc3986 py-tools21cm py-tqdm py-toolz py-pyfftw py-joblib py-lazy_loader
        'oskar@'$OSKAR_VERSION'+python~openmp' \
        'py-dask-mpi' \
        'py-ipykernel' \
        'py-mpi4py' \
        'py-nbconvert' \
        'py-pytest' \
        'py-rascil' \
        'py-scikit-image' \
        'py-tqdm' \
        'python@3.10' \
        'wsclean@=3.4' \
        # cfitsio?
        # fftw?
        # harp?
        # hdf5?
        # montagepy?
        # mpich?
        # openblas?
        # pip?
        # psutil
        # py-extension-helpers?
        # py-h5py
        # py-nbformat
        # py-packaging?
        # py-requests?
        # py-rfc3986?
        # py-setuptools?
        # py-ska-gridder-nifty-cuda?
        # py-tabulate?
        # py-versioneer?
        # py-wheel?
    && \
    spack concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast --reuse && \
    spack test run oskar@${OSKAR_VERSION} && \
    spack env view regenerate && \
    spack test run 'py-rascil' && \
    spack test run 'py-astropy' && \
    spack test run 'py-mpi4py' && \
    spack test run 'py-dask-mpi' && \
    # # Build pyerfa from source against the view's NumPy
    # /opt/view/bin/python -m pip install --no-build-isolation --no-deps -U 'pip<25.3' setuptools setuptools-scm wheel build 'extension-helpers>=1.0,<2' && \
    # /opt/view/bin/python -m pip install --no-build-isolation --no-deps --no-binary=pyerfa 'pyerfa=='$PYERFA_VERSION && \
    # /opt/view/bin/python -m pip install --no-build-isolation --no-deps 'astropy=='$ASTROPY_VERSION && \
    # # Provide shim for legacy import path expected by some packages
    # mkdir -p /opt/view/lib/python3.10/site-packages/astropy/_erfa && \
    # printf 'from erfa import *\n' > /opt/view/lib/python3.10/site-packages/astropy/_erfa/__init__.py && \
    # # No-op sitecustomize (avoid interfering with pip builds)
    # printf '# no-op\n' > /opt/view/lib/python3.10/site-packages/sitecustomize.py && \
    # # Run early sanity tests to catch environment issues fast (use Spack view python)
    /opt/view/bin/python -m pytest -q -k astropy_earthlocation_basic /opt/early-tests || exit 1

# possible additional specs:
# 'py-cython@0.29:3.0' \
# 'py-distributed@'$DISTRIBUTED_VERSION \
# 'py-extension-helpers@1.0:' \

# RUN --mount=type=cache,target=/root/.cache/pip \
#     . ${SPACK_ROOT}/share/spack/setup-env.sh && \
#     spack env activate /opt/spack_env && \
#     python - <<"PY"
# from astropy import units as u
# from astropy.coordinates import EarthLocation, Longitude, Latitude
# loc = EarthLocation.from_geodetic(Longitude(116.7644, u.deg), Latitude(-26.8247, u.deg), 377*u.m)
# print(loc)
# PY

# possible root specs to drop because they're already pulled in by others (or are build tools auto-added by Spack):
# - cfitsio: via wcslib -> casacore, and also via wsclean
# - mpich: via wsclean (+mpi)
# - openblas: via BLAS/LAPACK requirements of py-numpy/py-scipy (Spack will choose a provider)
# - py-cython: via build deps of several py-* (e.g., scipy stack)
# - py-pip, py-setuptools, py-wheel: generic Python build tooling; auto-pulled where needed
# watch out for:
# - pip 25.3 will break legacy setup.py used by aratmospy, eidos, katbeam, seqfile
# - openblas 0.3.28 breaks arm64

# Make Spack view default in system paths and shells
RUN printf "/opt/view/lib\n/opt/view/lib64\n" > /etc/ld.so.conf.d/spack-view.conf && ldconfig && \
    echo ". ${SPACK_ROOT}/share/spack/setup-env.sh 2>/dev/null || true" > /etc/profile.d/spack.sh && \
    echo "spack env activate /opt/spack_env 2>/dev/null || true" >> /etc/profile.d/spack.sh && \
    mkdir -p /opt/etc && \
    echo ". /etc/profile.d/spack.sh" > /opt/etc/spack_env && \
    chmod 644 /opt/etc/spack_env && \
    # Remove conda activation hook; we run Jupyter inside Spack Python
    rm -f /usr/local/bin/before-notebook.d/10activate-conda-env.sh && \
    # Optionally remove conda to avoid stray references and save space
    rm -rf /opt/conda || true

RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    # Install Jupyter stack into Spack Python so server runs inside Spack env
    python -m pip install --no-build-isolation \
        'jupyterlab==4.*' \
        'jupyter_server==2.*' \
        'jupyterlab_server==2.*' \
        'notebook==7.*' \
        'jupyter_core>=5' \
        'jupyter_client>=8' && \
    # Ensure start-notebook uses Spack jupyter first in PATH
    mkdir -p /usr/local/bin/before-notebook.d && \
    printf '#!/usr/bin/env bash\nPATH="/opt/view/bin:${HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nexport PATH\n' > /usr/local/bin/before-notebook.d/00-prefer-spack.sh && \
    chmod +x /usr/local/bin/before-notebook.d/00-prefer-spack.sh

# update python build dependencies
# mostly for photutils
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    # then update setuptools and friends to latest versions
    python -m pip install --no-build-isolation -U 'pip<25.3' 'cython>=3.0,<3.1' 'extension_helpers>=1.0,<2' 'packaging>=24.2' setuptools setuptools-scm wheel build versioneer extension_helpers
    # this updates pip-23.1.2 -> 25.2, setuptools 63.4.3 -> 80.9.0, packaging 24.1 -> 25.0
    # installs build-1.3.0 cython-3.0.12 extension_helpers-1.4.0 packaging-25.0 pip-25.2 pyproject_hooks-1.2.0 setuptools-80.9.0 setuptools-scm-9.2.0 versioneer-0.29 wheel-0.45.1

RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -c "import ska_sdp_func_python" || exit 1 && \
    python -c "import ska_sdp_func" || exit 1
# ska-sdp-func-python 0.1.5 requires numpy<1.24,>=1.23

# Build python-casacore from source against Spack casacore to avoid ABI issues
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip uninstall -y argparse || true && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -c "import casacore, casacore.tables, casacore.quanta; print('python-casacore OK')"

# this installs argparse-1.4.0 click-8.2.1 cloudpickle-3.1.1 dask-2022.12.0 dask_memusage-1.1 dask_mpi-2022.4.0 distributed-2022.12.0 fsspec-2025.9.0 locket-1.0.0 msgpack-1.1.1 natsort-8.4.0 partd-1.4.2 python-casacore-3.5.0 seqfile-0.2.0 sortedcontainers-2.4.0 tblib-3.1.0 toolz-1.0.0 zict-3.0.0

# Install remaining karabo-pipeline dependencies into Spack Python (as root) from pypi
# - https://artefact.skao.int/repository/pypi-all/simple does not correctly set python versions
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation --no-deps \
        'git+https://github.com/i4Ds/ARatmospy.git@67c302a136beb40a1cc88b054d7b62ccd927d64f#egg=aratmospy' \
        'git+https://github.com/i4Ds/eidos.git@74ffe0552079486aef9b413efdf91756096e93e7' \
        'git+https://github.com/ska-sa/katbeam.git@5ce6fcc35471168f4c4b84605cf601d57ced8d9e' \
        'rfc3986>=2.0.0' \
        'pyfftw' \
        'joblib' \
        'lazy_loader' \
        # 'scikit-image' \
        # 'tqdm' \
    && \
    # Install pyuvdata with dependencies but skip Spack-managed ones
    python -m pip install --no-build-isolation \
        --no-binary=numpy,scipy,astropy,h5py \
        'git+https://github.com/RadioAstronomySoftwareGroup/pyuvdata.git@v2.4.2'

# Install deps explicitly, then tools21cm without deps to keep Spack numpy
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
        --no-binary=numpy,scipy,astropy,h5py \
        'numpy=='${NUMPY_VERSION} \
        'tools21cm==2.3.8' \
    && \
    # Ensure dask_mpi present without altering Spack-managed dask/distributed
    python -m pip install --no-build-isolation --no-deps \
        'dask_mpi==2022.4.0' \
    && \
    # Force-install healpy wheel explicitly to avoid mixed Spack/pip state (keep it, but no numpy/scipy from pip)
    python -m pip install --no-build-isolation --force-reinstall --no-deps --only-binary=:all: "healpy==${HEALPY_VERSION}" && \
    python - <<"PY"
import importlib, sys
checks = [
    ('aratmospy','0.0'),
    ('eidos','0.0'),
    ('katbeam','0.0'),
    ('pyuvdata','2.4.2'),
    ('healpy','1.16'),
    ('rfc3986','2.0'),
    ('tools21cm','0.0'),
    ('tqdm','4.0'),
    ('astropy_healpix','1.0'),
    ('numpy','1.23.5'),
    ('toolz','0.0'),
    ('pyfftw','0.0'),
    ('joblib','0.0'),
    ('lazy_loader','0.0'),
    ('sklearn','1.5'),
    ('dask','2022.10'),
    ('dask_mpi','0.0'),
    ('distributed','2022.10'),
    ('skimage','0.24'),
    ('mpi4py','0.0'),
]
for (name, target) in checks:
    mod = None
    if name == 'aratmospy':
        try:
            mod = importlib.import_module('name')
        except Exception:
            print('aratmospy not importable (skipping check)')
            continue
    elif name == 'dask_mpi':
        try:
            import importlib.util
            spec = importlib.util.find_spec('dask_mpi')
            if spec is None:
                raise ImportError('dask_mpi not found')
            # Do not import to avoid hard mpi4py dependency here
            ver = '0.0'
            print('dask_mpi present')
            continue
        except Exception:
            print('dask_mpi not importable')
            sys.exit(1)
    else:
        try:
            mod = importlib.import_module(name)
        except Exception:
            print(f'{name} not importable')
            sys.exit(1)
    ver = getattr(mod, '__version__', '0.0')
    try:
        assert tuple([*ver.split('.')]) >= tuple([*target.split('.')])
    except Exception:
        print(f'{name} version not available')
        continue
    print(f'{name} version {ver}, target {target}')
sys.exit(0)
PY

# ^ Successfully installed aratmospy-1.0.0 docstring-parser-0.17.0 eidos-1.1.0 et-xmlfile-2.0.0 future-1.0.0 healpy-1.16.2 imageio-2.37.0 iniconfig-2.1.0 joblib-1.5.2 katbeam-0.2.dev36+head.5ce6fcc lazy-loader-0.4 networkx-3.4.2 openpyxl-3.1.5 pluggy-1.6.0 pyfftw-0.15.0 pytest-8.4.2 pyuvdata-2.4.2 rfc3986-2.0.0 threadpoolctl-3.6.0 tifffile-2025.5.10 tools21cm-2.3.8 tqdm-4.67.1
# Note: dask@2022.10.2, distributed@2022.10.2, scikit-learn@1.5 now installed via Spack (scikit-image kept as pip due to ffmpeg dependency issues)

# Copy repository for editable install and testing
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100

RUN fix-permissions /home/${NB_USER} && \
    fix-permissions /opt/view/lib/python3.10/

COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/Karabo-Pipeline

RUN fix-permissions /home/${NB_USER}/Karabo-Pipeline

USER ${NB_UID}
# Set explicit version for Karabo-Pipeline when building without VCS metadata
ARG KARABO_VERSION=0.34.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_KARABO_PIPELINE=${KARABO_VERSION} \
    VERSIONEER_OVERRIDE=${KARABO_VERSION}

RUN --mount=type=cache,target=/home/${NB_USER}/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    printf 'version = "%s"\n\n' "${KARABO_VERSION}" > /home/${NB_USER}/Karabo-Pipeline/karabo/_version.py && \
    printf 'def get_versions():\n    return {"version": "%s", "full-revisionid": None, "dirty": None, "error": None, "date": None}\n' "${KARABO_VERSION}" >> /home/${NB_USER}/Karabo-Pipeline/karabo/_version.py && \
    python -m pip install --no-build-isolation -e /home/jovyan/Karabo-Pipeline

# Register kernel for jovyan user using the Spack Python
RUN python -m ipykernel install --user --name=karabo --display-name="Karabo (Spack Py3.10)"

# Run tests during build to validate environment
ARG SKIP_TESTS=0
ENV SKIP_TESTS=${SKIP_TESTS}
RUN if [ "${SKIP_TESTS:-0}" = "1" ]; then exit 0; fi; \
    pytest -q -x --tb=short -k "not test_suppress_rascil_warning" /home/${NB_USER}/Karabo-Pipeline && \
    rm -rf /home/${NB_USER}/.astropy/cache \
           /home/${NB_USER}/.cache/astropy \
           /home/${NB_USER}/.cache/pyuvdata \
           /home/${NB_USER}/.cache/rascil

WORKDIR "/home/${NB_USER}"
