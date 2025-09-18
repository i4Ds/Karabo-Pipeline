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

# Overlay repo with legacy py-xarray version compatible with rascil/ska-sdp
COPY spack-overlay /opt/karabo-spack
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/karabo-spack

ARG NUMPY_VERSION=1.23.5
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
# distributed needed by rascil, dask
ARG DISTRIBUTED_VERSION=2022.12

# first install xarray 2022.12.0
# requires:
# boto 1.18 1.20
# cartopy 0.19 0.20
# distributed 2021.09 2021.11
# dask 2021.09 2021.11
# h5py 3.1 3.6
# hdf5 1.10 1.12
# matplotlib-base 3.4 3.5
# nc-time-axis 1.3 1.4
# netcdf4 1.5.3 1.5.7
# packaging 20.3 21.3
# pint 0.17 0.18
# pseudonetcdf 3.1 3.2
# typing_extensions 3.10 4.0
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
        'py-pandas@'$PANDAS_VERSION \
        'py-pip@:25.2' \
        'py-xarray@'$XARRAY_VERSION \
        'python@3.10' \
    && \
    spack concretize --force && \
    # ac_cv_lib_curl_curl_easy_init=no for cfitsio
    spack install --no-check-signature --no-checksum --fail-fast && \
    spack env view regenerate

ARG SCIPY_VERSION=1.9.3
# scipy needed by pyuvdata bluebild rascil scikit-image astroml ska-sdp-func-python aratmospy bdsf reproject tools21cm gwcs photutils healpy scikit-learn eidos
# scipy 1.13.1 installed by conda
ARG MATPLOTLIB_VERSION=3.6
# matplotlib needed by bluebild rascil aratmospy tools21cm
ARG ASTROPY_VERSION=5.1
# astropy needed by rascil pyuvdata ska-sdp-func-python aratmospy bdsf tools21cm gwcs photutils ska-sdp-datamodels healpy eidos bluebild
ARG CASACORE_VERSION=3.5.0
# casacore needed by everybeam wsclean oskar rascil

# Create Spack environment, enable view at /opt/view, add packages, install
# ac_cv_lib_curl_curl_easy_init=no is needed for cfitsio
RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    spack env activate /opt/spack_env; \
    spack add \
        'boost+python+numpy' \
        'casacore@='$CASACORE_VERSION' +python' \
        'mpich' \
        'openblas@:0.3.27' \
        'py-h5py@'$H5PY_VERSION \
        'hdf5@'$HDF5_VERSION \
        # 'py-distributed@'$DISTRIBUTED_VERSION \
        # 'py-astropy@'$ASTROPY_VERSION \
        # 'py-healpy' \
        'py-ipykernel' \
        'py-matplotlib@'$MATPLOTLIB_VERSION \
        'py-mpi4py' \
        'py-nbconvert' \
        'py-requests' \
        'py-scipy@'$SCIPY_VERSION \
        'py-tabulate' \
        'py-tqdm' \
        'wsclean@=3.4' \
    && \
    spack concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast && \
    spack env view regenerate

# possible additional specs:
# 'py-cython' \
# 'py-extension-helpers' \
# 'py-pytest' \
# 'py-setuptools' \
# 'py-versioneer' \
# 'py-wheel' \

# issues:
# - py-bdsf does not set version metadata correctly, breaks later pip installs
# - py-casacore@3.5.0 is totally broken, pip installing python-casacore==3.5.0 works instead
# - only py-photutils@1.5.0 is available from spack, but @1.11.0 needed by rascil@1.0.0
#   - <https://github.com/spack/spack-packages/blob/develop/repos/spack_repo/builtin/packages/py_photutils/package.py>
# - only py-reproject@0.7.1 is available from spack, but @0.9.1 needed by ska-sdp-datamodels@0.1.3:
#   - <https://github.com/spack/spack-packages/blob/develop/repos/spack_repo/builtin/packages/py_reproject/package.py>
# - py-astropy@4 is being installed, rascil needs @5.1 which is available
#   - used by py-bdsf, py-healpy, (py-photutils, py-reproject) which do not place any version constraints
# - only ska-sdp-datamodels@0.3.3 is available from sdp spack, but we need 0.1.3 for ska-sdp-func-python
# - only ska-sdp-func-python@0.5.1 is available from sdp spack, but we need 0.1.5 for rascil@1.0.0
# - only ducc0@0.34:0.36 is available from sdp spack, but we need 0.27 for ska-sdp-func-python
# possible root specs to drop because they're already pulled in by others (or are build tools auto-added by Spack):
# - cfitsio: via wcslib -> casacore, and also via wsclean
# - mpich: via wsclean (+mpi)
# - openblas: via BLAS/LAPACK requirements of py-numpy/py-scipy (Spack will choose a provider)
# - py-extension-helpers: via py-astropy (dep of py-bdsf, py-healpy)
# - py-cython: via build deps of several py-* (e.g., scipy stack)
# - py-numpy: via py-pandas, py-scipy, py-h5py
# - py-pip, py-setuptools, py-wheel: generic Python build tooling; auto-pulled where needed
# - py-pybind11: only needed if a root package actually uses it; otherwise drop
# Possibly unnecessary (not clear they're required transitively by any other root spec):
# - py-matplotlib, py-nbconvert, py-requests, py-tabulate, py-tqdm, py-versioneer, py-ipykernel, py-h5py
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

# Prefer Spack view in PATH, and ensure batch shells source spack env
ENV PATH="/opt/view/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64:${LD_LIBRARY_PATH}" \
    BASH_ENV=/opt/etc/spack_env \
    PYTHONNOUSERSITE=1 \
    CMAKE_PREFIX_PATH="/opt/view:${CMAKE_PREFIX_PATH}" \
    PKG_CONFIG_PATH="/opt/view/lib/pkgconfig:/opt/view/lib64/pkgconfig:${PKG_CONFIG_PATH}"

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
    printf '#!/usr/bin/env bash\nPATH="/opt/view/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nexport PATH\n' > /usr/local/bin/before-notebook.d/00-prefer-spack.sh && \
    chmod +x /usr/local/bin/before-notebook.d/00-prefer-spack.sh

# update python build dependencies
# mostly for photutils
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    # then update setuptools and friends to latest versions
    python -m pip install --no-build-isolation -U 'pip<25.3' 'cython>=3.0,<3.1' 'extension_helpers>=1.0,<2' 'packaging>=24.2' setuptools setuptools-scm wheel build versioneer extension_helpers
    # this updates pip-23.1.2 -> 25.2, setuptools 63.4.3 -> 80.9.0, packaging 24.1 -> 25.0
    # and installs build-1.3.0 cython-3.0.12 extension_helpers-1.4.0 packaging-25.0 pip-25.2 pyproject_hooks-1.2.0 setuptools-80.9.0 setuptools-scm-9.2.0 tomli-2.2.1 versioneer-0.29 wheel-0.45.1

ARG OSKAR_VERSION=2.8.3
# karabo uses 2.8.3.
# 2.10.0 works on arm64 but gives code -115 when reading vis files
# oskarpy not available on pip
# 'git+https://github.com/OxfordSKA/OSKAR.git@2.8.3#egg=oskarpy&subdirectory=python'
ENV OSKAR_INC_DIR=/opt/software/include \
    OSKAR_LIB_DIR=/opt/software/lib
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    git clone 'https://github.com/OxfordSKA/OSKAR.git' /opt/oskar && \
    cd /opt/oskar/ && \
    git checkout $OSKAR_VERSION && \
    mkdir build && \
    cd build && \
    LD_SYS="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/lib:/usr/lib" && \
    env LD_LIBRARY_PATH="${LD_SYS}:/opt/view/lib:/opt/view/lib64" cmake -DCMAKE_INSTALL_PREFIX=/opt/software -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/view .. && \
    # too many threads crashes arm64
    env LD_LIBRARY_PATH="${LD_SYS}:/opt/view/lib:/opt/view/lib64" make -j && \
    env LD_LIBRARY_PATH="${LD_SYS}:/opt/view/lib:/opt/view/lib64" make install && \
    python -m pip install --no-build-isolation '/opt/oskar/python'
    # python -c "pkg=__import__('oskar'); target='$OSKAR_VERSION'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])"

ARG ASTROPLAN_VERSION=0.8
# astroplan needed by ska-sdp-datamodels (and ska-sdp-func-python)
# astroplan 0.10.1 installed by conda
# astroplan 0.8 needed by ska-sdp-datamodels 0.1.3
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPLAN=${ASTROPLAN_VERSION} && \
    python -m pip install --no-build-isolation \
        'astropy=='${ASTROPY_VERSION} && \
    python -m pip install --no-build-isolation --no-deps \
        'git+https://github.com/astropy/astroplan.git@v'${ASTROPLAN_VERSION} && \
    python -c "pkg=__import__('astroplan'); target='${ASTROPLAN_VERSION}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])"

# ska-sdp-datamodels, required by ska-sdp-func-python (and rascil)
# - Install without dependency resolution, version checks don't work.
# - install without deps because ... ?
# - ska-sdp-datamodels 0.1.3 requires xarray<2023.0.0,>=2022.10.0, but you have xarray 2023.7.0 which is incompatible.
ARG SKA_SDP_DATAMODELS_VERSION=0.1.3
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
        --index-url=https://artefact.skao.int/repository/pypi-all/simple \
        --extra-index-url=https://pypi.org/simple \
        'h5py=='$H5PY_VERSION \
        'numexpr==2.10.2' \
        'xarray=='$XARRAY_VERSION \
    && \
    for pkg_ver in \
        astroplan:$ASTROPLAN_VERSION astropy:5.1 \
        h5py:3.7.0 numpy:1.23.4 pandas:1.5 xarray:2022.10.0 \
        packaging:21.3 numexpr:2.10.2; \
    do \
        python -c "pkg=__import__('${pkg_ver%:*}'); target='${pkg_ver#*:}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])" || exit 1 ; \
    done && \
    python -m pip install --no-build-isolation \
        --index-url=https://artefact.skao.int/repository/pypi-all/simple \
        --extra-index-url=https://pypi.org/simple \
        'ska-sdp-datamodels=='$SKA_SDP_DATAMODELS_VERSION \
        && \
    python -c "import ska_sdp_datamodels" || exit 1

ARG PHOTUTILS_VERSION=1.11.0
# photutils needed by ska-sdp-func-python
# photutils 1.8.0 installed by conda
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    for pkg_ver in \
        cython:3.0 extension_helpers:1.0 \
        numpy:1.22 astropy:5.1 \
    ; do \
        python -c "pkg=__import__('${pkg_ver%:*}'); target='${pkg_ver#*:}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])" || exit 1 ; \
    done && \
    python -c 'import setuptools_scm' || exit 1 && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PHOTUTILS=${PHOTUTILS_VERSION} && \
    python -m pip install --no-build-isolation 'git+https://github.com/astropy/photutils.git@'${PHOTUTILS_VERSION} && \
    python -c "pkg=__import__('photutils'); target='${PHOTUTILS_VERSION}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])"

ARG DUCC0_VERSION=0.27
# ducc0 needed by ska-sdp-func-python
# ducc0 0.27.0 installed by conda
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation 'ducc0=='$DUCC0_VERSION && \
    python -c "pkg=__import__('ducc0'); target='${DUCC0_VERSION}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])"

ARG SKA_SDP_FUNC_PYTHON_VERSION=0.1.5
# rascil needs ska-sdp-func-python
# ska-sdp-func-python 0.1.4 installed by conda
# ska-sdp-func-python 0.1.5 needs astroplan >=0.8, astropy >=5.1, ducc0 >=0.27.0,<0.28, numpy >=1.26.4,<2.0a0, photutils >=1.5, scipy >=1.10.1, ska-sdp-datamodels 0.1.3.*, ska-sdp-func 0.0.6.*, xarray 2023.2.*
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation --no-deps \
        --index-url=https://artefact.skao.int/repository/pypi-all/simple \
        --extra-index-url=https://pypi.org/simple \
        'ska-sdp-func-python=='$SKA_SDP_FUNC_PYTHON_VERSION \
        'git+https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git@08eb17cf9f4d63320dd0618032ddabc6760188c9' \
        && \
    python -c "import ska_sdp_func_python" || exit 1 && \
    python -c "import ska_sdp_func" || exit 1
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# ska-sdp-func-python 0.1.5 requires numpy<1.24,>=1.23, but you have numpy 1.25.2 which is incompatible.
# Successfully installed asciitree-0.3.3 astropy-healpix-1.1.2 fasteners-0.20 numcodecs-0.13.1 numpy-1.25.2 reproject-0.14.1 zarr-2.18.3

ARG BDSF_VERSION=1.12.0
# bdsf needed by rascil
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    # Pin legacy build toolchain for PyBDSF metadata/f2py to succeed
    # scikit-build is required by newer PyBDSF builds (provides 'skbuild')
    python -m pip install --no-build-isolation --upgrade \
        'setuptools>=64' 'Cython<3' 'numpy==1.23.5' 'scikit-build' && \
    for pkg_ver in \
        astropy:0.0 numpy:0.0 scipy:0.0 skbuild:0.0 \
    ; do \
        python -c "pkg=__import__('${pkg_ver%:*}'); target='${pkg_ver#*:}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])" || exit 1 ; \
    done && \
    # Force using stdlib distutils so numpy.distutils can import distutils.*
    export SETUPTOOLS_USE_DISTUTILS=stdlib && \
    # Try to use a prebuilt wheel first; if unavailable, build from source with setuptools_scm
    ( python -m pip install --no-build-isolation --only-binary=:all: 'bdsf=='${BDSF_VERSION} \
      && python -c "import bdsf" ) \
    || ( \
      python -m pip install --no-build-isolation 'setuptools_scm>=8' && \
      export SETUPTOOLS_SCM_PRETEND_VERSION=${BDSF_VERSION} && \
      export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BDSF=${BDSF_VERSION} && \
      python -m pip install --no-build-isolation 'git+https://github.com/lofar-astron/PyBDSF.git@v'${BDSF_VERSION} && \
      python -c "import bdsf" \
    )
    # python -c "pkg=__import__('bdsf'); target='${BDSF_VERSION}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])"

ARG DASK_VERSION=2022.12
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
        'dask_memusage>=1.1' \
        'dask_mpi' \
        'dask=='${DASK_VERSION} \
        'distributed=='${DISTRIBUTED_VERSION} \
        'seqfile>=0.2.0' \
        'tabulate>=0.9' \
    || exit 1
# Build python-casacore from source against Spack casacore to avoid ABI issues
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip uninstall -y argparse || true && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    export CMAKE_PREFIX_PATH="/opt/view:${CMAKE_PREFIX_PATH}" && \
    export CASACORE_ROOT=/opt/view && \
    export CPATH="/opt/view/include:${CPATH}" && \
    export LIBRARY_PATH="/opt/view/lib:/opt/view/lib64:${LIBRARY_PATH}" && \
    export LD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64:${LD_LIBRARY_PATH}" && \
    python -m pip install --no-build-isolation --no-binary=:all: \
        'python-casacore=='${CASACORE_VERSION} && \
    python -c "import casacore, casacore.tables, casacore.quanta; print('python-casacore OK')"

# this installs argparse-1.4.0 click-8.2.1 cloudpickle-3.1.1 dask-2022.12.0 dask_memusage-1.1 dask_mpi-2022.4.0 distributed-2022.12.0 fsspec-2025.9.0 locket-1.0.0 msgpack-1.1.1 natsort-8.4.0 partd-1.4.2 python-casacore-3.5.0 seqfile-0.2.0 sortedcontainers-2.4.0 tblib-3.1.0 toolz-1.0.0 zict-3.0.0

ARG REPROJECT_VERSION=0.9.1
# reproject needed by rascil, requires dask, numpy, scipy, shapely, zarr
# reproject 0.14.1 installed by conda
# reproject 0.9 required by rascil 1.0.0
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip uninstall -y argparse || true && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_REPROJECT=${REPROJECT_VERSION} && \
    export SETUPTOOLS_USE_DISTUTILS=stdlib && \
    # Install reproject without pulling newer deps; rely on pinned numpy/scipy already present
    python -m pip install --no-build-isolation --no-deps \
        'astropy-healpix==1.0.0' \
        'git+https://github.com/astropy/reproject.git@v'${REPROJECT_VERSION} && \
    python -c "pkg=__import__('reproject'); target='${REPROJECT_VERSION}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])"
    # keep reproject <0.10 to satisfy rascil 1.0.0

# rascil (optional; disabled by default due to dependency conflicts)
# https://artefact.skao.int/service/rest/repository/browse/pypi-all/rascil/
# this reinstalls pip 22.1.2 -> 25.2, wheel 0.37.1 -> 0.45.1
ARG RASCIL_VERSION=1.0.0
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    if [ "${RASCIL_VERSION:-0}" = "0" ]; then \
        echo "Skipping rascil install"; \
        exit 0; \
    fi; \
    for pkg_ver in \
        numpy:$NUMPY_VERSION scipy:$SCIPY_VERSION pandas:$PANDAS_VERSION xarray:$XARRAY_VERSION \
        dask:$DASK_VERSION distributed:$DISTRIBUTED_VERSION astropy:$ASTROPY_VERSION matplotlib:$MATPLOTLIB_VERSION \
        h5py:$H5PY_VERSION reproject:0.9 casacore:$CASACORE_VERSION \
        seqfile:0.2 tabulate:0.9 dask_memusage:1.1; \
    do \
        python -c "pkg=__import__('${pkg_ver%:*}'); target='${pkg_ver#*:}'; print(f'{pkg.__name__} installed {pkg.__version__}, target {target}'); assert tuple([*pkg.__version__.split('.')]) >= tuple([*target.split('.')])" || exit 1; \
    done; \
    python -c 'import ska_sdp_datamodels' || exit 1 && \
    python -c 'import ska_sdp_func_python' || exit 1 && \
    python -m pip install --no-build-isolation --no-deps \
        --index-url=https://artefact.skao.int/repository/pypi-all/simple \
        --extra-index-url=https://pypi.org/simple \
        'rascil=='$RASCIL_VERSION && \
    python -c "import rascil" || exit 1;

# Install remaining karabo-pipeline dependencies into Spack Python (as root) from pypi
# - https://artefact.skao.int/repository/pypi-all/simple does not correctly set python versions
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
        'git+https://github.com/i4Ds/ARatmospy.git@67c302a136beb40a1cc88b054d7b62ccd927d64f#egg=aratmospy' \
        'git+https://github.com/i4Ds/eidos.git@74ffe0552079486aef9b413efdf91756096e93e7' \
        'git+https://github.com/ska-sa/katbeam.git@5ce6fcc35471168f4c4b84605cf601d57ced8d9e' \
        'pyuvdata==2.4.2' \
        'healpy==1.16.2' \
        'rfc3986>=2.0.0' \
        'tools21cm' \
        'scipy=='${SCIPY_VERSION} \
        'numpy=='${NUMPY_VERSION} \
        'astropy-healpix==1.0.0' \
    && \
    python - <<'PY'
import importlib, sys
checks = [
    ('aratmospy','0.0'),
    ('eidos','0.0'),
    ('katbeam','0.0'),
    ('pyuvdata','2.4.2'),
    ('healpy','1.16'),
    ('rfc3986','2.0'),
    ('tools21cm','0.0'),
    ('astropy_healpix','1.0'),
]
for (name, target) in checks:
    try:
        if name == 'aratmospy':
            try:
                mod = importlib.import_module('aratmospy')
            except Exception:
                mod = importlib.import_module('ARatmospy')
        else:
            mod = importlib.import_module(name)
    except Exception:
        print(f'{name} not importable')
        sys.exit(1)
    try:
        ver = getattr(mod, '__version__', '0.0')
        assert tuple([*ver.split('.')]) >= tuple([*target.split('.')])
    except Exception:
        print(f'{name} version not available')
        sys.exit(1)
    print(f'{name} version {ver}, target {target}')
sys.exit(0)
PY

# ^ Successfully installed aratmospy-1.0.0 docstring-parser-0.17.0 eidos-1.1.0 et-xmlfile-2.0.0 future-1.0.0 healpy-1.16.2 imageio-2.37.0 iniconfig-2.1.0 joblib-1.5.2 katbeam-0.2.dev36+head.5ce6fcc lazy-loader-0.4 networkx-3.4.2 openpyxl-3.1.5 pluggy-1.6.0 pyfftw-0.15.0 pytest-8.4.2 pyuvdata-2.4.2 rfc3986-2.0.0 scikit-image-0.24.0 scikit-learn-1.7.2 threadpoolctl-3.6.0 tifffile-2025.5.10 tools21cm-2.3.8 tqdm-4.67.1

# Copy repository for editable install and testing
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100
COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/Karabo-Pipeline

RUN fix-permissions /home/${NB_USER} && \
    fix-permissions /home/${NB_USER}/Karabo-Pipeline && \
    fix-permissions /opt/view/lib/python3.10/
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
RUN if [ "${SKIP_TESTS:-0}" = "1" ]; then exit 0; fi; pytest -q -x --tb=short -k "not test_suppress_rascil_warning" /home/${NB_USER}/Karabo-Pipeline

WORKDIR "/home/${NB_USER}"
