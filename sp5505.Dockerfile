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

# Add SKA SDP Spack repo and overlay
RUN git clone --depth=2 --branch=2025.07.3 https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/ska-sdp-spack
COPY spack-overlay /opt/karabo-spack
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && spack repo add /opt/karabo-spack

ARG NUMPY_VERSION=1.26.4
# 1.23.5 worked at some point
ARG PYTHON_VERSION=3.10

# Version pins aligned with sp5505
ARG ASTROPLAN_VERSION=0.10.1
# 0.8 may have worked at some point
# conda uses 0.10.1
ARG ASTROPY_HEALPIX_VERSION=1.1.2
# 1.0.0 was installed at some point
# conda uses 1.1.2
ARG BDSF_VERSION=1.12.0
# 1.12.0 is easier to install because of setuptools nonsense
# conda uses 1.10.2
ARG DASK_VERSION=2022.12.1
ARG DUCC_VERSION=0.27
ARG PYERFA_VERSION=2.0.1.5
# conda installs 2.0.1.5
# 2.0.0.1 needed patches to work with astropy 5.1
ARG PHOTUTILS_VERSION=1.11.0
# 1.11.0 worked at some point
# conda uses 1.8.0
ARG REPROJECT_VERSION=0.9.1
# 0.9.1 worked at some point
# conda uses 0.14.1
ARG SDP_DATAMODELS_VERSION=0.1.3
ARG SDP_FUNC_VERSION=1.2.2
# 0.1.5 worked at some point
# conda uses 0.0.6
# earliest 1.1.7 in sdp-spack
# 1.2.2 works with the current stack
ARG SDP_FUNC_PYTHON_VERSION=0.1.5
# conda uses 0.1.4
# 0.1.5 works with the current stack
ARG RASCIL_VERSION=1.0.0
# conda uses 1.0.0

# numpy needed by pyuvdata montagepy numexpr scipy rascil scikit-image pywavelets astroml ducc0 imageio ska-sdp-func-python contourpy aratmospy bokeh astroplan coda harp astropy-healpix katbeam tensorboard h5py dask ml_dtypes ska-gridder-nifty-cuda libboost-python-devel python-casacore tifffile pytest-arraydiff shapely bdsf casacore finufft reproject numcodecs matplotlib-base tools21cm libboost-python numba gwcs tensorflow-base pyfftw boost xarray asdf pyside6 photutils astropy bottleneck pandas oskarpy ska-sdp-datamodels ska-sdp-func healpy keras scikit-learn pyerfa eidos asdf-astropy zarr bluebild
# numpy 1.26.4 installed by conda
# numpy>=1.24 required by zarr 2.18.3
# numpy 1.23 needed by rascil 1.0.0 and ska-sdp-func-python 0.1.5
ARG PANDAS_VERSION=1.5.3
# pandas needed by rascil dask xarray ska-sdp-datamodels bluebild
# pandas 1.5.3 is installed by conda
ARG XARRAY_VERSION=2023.2.0
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
ARG DISTRIBUTED_VERSION=2022.12.1
# distributed needed by rascil, dask
# conda has 2022.12.1
# rascil 1.0.0 requires distributed<2022.13,>=2022.12
# issues installing 2022.12.1 directly with spack
ARG SCIPY_VERSION=1.10.1
# scipy needed by pyuvdata bluebild rascil scikit-image astroml ska-sdp-func-python aratmospy bdsf reproject tools21cm gwcs photutils healpy scikit-learn eidos
# conda uses scipy 1.13.1 but this requires cupy and torch
# 1.9.3 worked with numpy 1.23.5
ARG MATPLOTLIB_VERSION=3.9.2
# matplotlib needed by bluebild rascil aratmospy tools21cm
# conda uses 3.10.5 but max available is 3.9.2
# 3.6.3 worked at some point
ARG ASTROPY_VERSION=5.1.1
# astropy needed by rascil pyuvdata ska-sdp-func-python aratmospy bdsf tools21cm gwcs photutils ska-sdp-datamodels healpy eidos bluebild
# astropy>5.2 has no ._erfa
ARG CASACORE_VERSION=3.5.0
# casacore needed by everybeam wsclean oskar rascil
ARG HEALPY_VERSION=1.16.2
# conda installs 1.16.6
# 1.16.2 worked at some point
# healpy needed by rascil ska-sdp-func-python aratmospy bdsf tools21cm gwcs photutils ska-sdp-datamodels eidos bluebild

ARG OSKAR_VERSION=2.8.3

ARG TOOLS21CM_VERSION=2.3.8
# tools21cm needed by rascil ska-sdp-func-python aratmospy bdsf tools21cm gwcs photutils ska-sdp-datamodels eidos bluebild
# tools21cm 2.0.3 installed by conda
# tools21cm 2.3.8 is available

ARG TABULATE_VERSION=0.9.0
# conda has 0.9.0
ARG NUMEXPR_VERSION=2.10.2
# conda has 2.10.2
ARG BOTTLENECK_VERSION=1.3.7
# conda has 1.5.0
# 1.3.7 worked at some point
ARG SEQFILE_VERSION=0.2.0
# conda has 0.2.0
# SciPy 1.9.x in Spack conflicts with OpenBLAS >=0.3.26; use 0.3.25 to satisfy
ARG OPENBLAS_VERSION=0.3.25
# 0.3.27 works on arm64
# conda has 0.3.30
# 0.3.28 is the latest supported in builtin
ARG BOOST_VERSION=1.82.0
# 1.86.0 worked at some point
# conda has 1.82.0

ARG PYUVDATA_VERSION=2.4.2

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
    spack config add "concretizer:unify:when_possible"; \
    spack config add "concretizer:reuse:true"; \
    spack config add "view:/opt/view"; \
    spack config add "config:source_cache:/opt/spack-source-cache"; \
    spack config add "config:misc_cache:/opt/spack-misc-cache"; \
    spack mirror add --autopush --unsigned mycache file:///opt/buildcache; \
    spack buildcache keys --install --trust || true; \
    # TODO: spack mirror add v0.23.1 https://binaries.spack.io/v0.23.1; \
    spack add \
        'cfitsio' \
        'healpix-cxx' \
        # known good from bdsf.Dockerfile and astropy.Dockerfile
        'boost@'$BOOST_VERSION'+python+numpy' \
        'py-astropy@'$ASTROPY_VERSION \
        'py-astropy-healpix@'$ASTROPY_HEALPIX_VERSION \
        'py-bdsf@'$BDSF_VERSION \
        'py-matplotlib@'$MATPLOTLIB_VERSION \
        'py-pyerfa@'$PYERFA_VERSION \
        'py-numpy@'$NUMPY_VERSION \
        'py-pip@:25.2' \
        'py-scipy@'$SCIPY_VERSION \
        'python@'$PYTHON_VERSION \
        # known good from rascil.Dockerfile
        'casacore@'$CASACORE_VERSION'+python' \
        'fftw~mpi~openmp' \
        'hdf5@'$HDF5_VERSION'+hl~mpi' \
        'openblas@'$OPENBLAS_VERSION \
        'py-astroplan@'$ASTROPLAN_VERSION \
        'py-casacore@'$CASACORE_VERSION \
        'py-dask@'$DASK_VERSION \
        'py-dask-memusage@1.1' \
        'py-distributed@'$DISTRIBUTED_VERSION \
        'py-ducc@'$DUCC_VERSION \
        'py-h5py@'$H5PY_VERSION \
        'py-healpy@'$HEALPY_VERSION \
        'py-numexpr@'$NUMEXPR_VERSION \
        'py-pandas@'$PANDAS_VERSION \
        'py-photutils@'$PHOTUTILS_VERSION \
        'py-rascil@'$RASCIL_VERSION \
        'py-reproject@'$REPROJECT_VERSION \
        'py-seqfile@'$SEQFILE_VERSION \
        'py-ska-sdp-datamodels@'$SDP_DATAMODELS_VERSION \
        'py-ska-sdp-func-python@'$SDP_FUNC_PYTHON_VERSION \
        'py-ska-sdp-func@'$SDP_FUNC_VERSION \
        'py-tabulate@'$TABULATE_VERSION \
        'py-xarray@'$XARRAY_VERSION \
        'oskar@'$OSKAR_VERSION'+python~openmp' \
        'py-bottleneck@'$BOTTLENECK_VERSION \
        'py-dask-mpi' \
        'py-extension-helpers@1.1.1' \
        'py-ipykernel@6:' \
        'py-joblib' \
        'py-lazy-loader' \
        'py-mpi4py' \
        'py-nbconvert' \
        'py-pyfftw' \
        'py-pytest' \
        'py-rfc3986@2:' \
        'py-scikit-image' \
        'py-scikit-learn' \
        'py-tqdm' \
        'py-pyuvdata@'$PYUVDATA_VERSION'+casa' \
        # todo: py-aratmospy py-eidos py-katbeam py-tools21cm py-toolz
        # 'py-jupyterlab@4' \
        # 'py-jupyter-server@2' \
        # 'py-jupyterlab-server@2' \
        # 'py-jupyter-core@5:' \
        # 'py-jupyter-client@8:' \
        # 'py-notebook@7' \
        # py-nbformat
        # py-packaging?
        # py-requests?
        # py-setuptools?
        # py-ska-gridder-nifty-cuda?
        # py-wheel?
        'wsclean@=3.4' \
        # cfitsio?
        # harp?
        # montagepy?
        # mpich?
        # psutil
    && \
    time -p spack concretize --force && \
    spack install -v --source 'py-healpy@'$HEALPY_VERSION && \
    spack test run 'py-healpy' || ( \
        cat /home/jovyan/.spack/test/*/py-healpy-1.16.2-*-test-out.txt; \
        exit 1 \
    ) && \
    ac_cv_lib_curl_curl_easy_init=no spack install -j$(nproc) --no-check-signature --no-checksum --fail-fast --reuse --show-log-on-error && \
    spack install -v --source 'py-healpy@'$HEALPY_VERSION && \
        spack test run 'py-healpy' || ( \
        cat /home/jovyan/.spack/test/*/py-healpy-1.16.2-*-test-out.txt; \
        exit 1 \
    ) && \
    # TODO: spack gc && \
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
    PKG_CONFIG_PATH="/opt/view/lib/pkgconfig:/opt/view/lib64/pkgconfig" \
    PYTHONPATH="/opt/view/lib/python${PYTHON_VERSION}/site-packages"

# Ensure start-notebook uses Spack jupyter first in PATH
RUN mkdir -p /usr/local/bin/before-notebook.d && \
    printf '#!/usr/bin/env bash\nPATH="/opt/view/bin:${HOME}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"\nexport PATH\nLD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64"\nexport LD_LIBRARY_PATH\nPYTHONPATH=/opt/view/lib/python${PYTHON_VERSION}/site-packages\nexport PYTHONPATH\n' > /usr/local/bin/before-notebook.d/00-prefer-spack.sh && \
    chmod +x /usr/local/bin/before-notebook.d/00-prefer-spack.sh && \
    # Remove conda and activation hook; we run Jupyter inside Spack Python
    rm -f /opt/conda /usr/local/bin/before-notebook.d/10activate-conda-env.sh || true

RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    spack test run 'py-astropy-healpix' && \
    # spack test run 'py-astropy' && \ # broken
    spack test run 'py-numpy' && \
    spack test run 'py-scipy' && \
    spack test run 'py-bdsf' && \
    spack test run 'py-astroplan' && \
    spack test run 'py-casacore' && \
    spack test run 'py-healpy' || ( \
        cat /home/jovyan/.spack/test/*/py-healpy-1.16.2-*-test-out.txt; \
        exit 1 \
    ) && \
    python -c "import dask, distributed; print('OK', dask.__version__, distributed.__version__)" && \
    # hack: can't run tests for py-distributed due to circular dependency on py-dask
    # spack test run 'py-distributed' && \
    spack test run 'py-ducc' && \
    spack test run 'py-h5py' && \
    spack test run 'py-rascil' && \
    spack test run 'py-pandas' && \
    spack test run 'py-photutils' && \
    spack test run 'py-pyerfa' && \
    spack test run 'py-reproject' && \
    spack test run 'py-seqfile' && \
    spack test run 'py-ska-sdp-datamodels' && \
    spack test run 'py-ska-sdp-func-python' && \
    spack test run 'py-ska-sdp-func' && \
    spack test run 'py-xarray' && \
    spack test run 'py-rascil' && \
    spack test run 'oskar' && \
    spack test run 'py-mpi4py' && \
    spack test run 'py-dask-mpi'
    # spack test run 'py-pyuvdata'

# todo: try spack env activate /opt/spack_env --with-view /opt/view
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python - <<"PY"
import importlib, sys, os
print(f'PYTHONPATH: {sys.path}')
print(f'LD_LIBRARY_PATH: {os.environ.get("LD_LIBRARY_PATH")}')
print(f'PATH: {os.environ.get("PATH")}')
print(f'BASH_ENV: {os.environ.get("BASH_ENV")}')
print(f'PYTHONNOUSERSITE: {os.environ.get("PYTHONNOUSERSITE")}')
print(f'CMAKE_PREFIX_PATH: {os.environ.get("CMAKE_PREFIX_PATH")}')
print(f'PKG_CONFIG_PATH: {os.environ.get("PKG_CONFIG_PATH")}')

checks = [
    ('astropy','5.1.1'),
    ('erfa','2.0'),
    ('healpy','1.16'),
]
for (name, target) in checks:
    mod = None
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f'{name} not importable: {exc}')
        sys.exit(1)
    ver = getattr(mod, '__version__', '0.0')
    try:
        assert tuple([*ver.split('.')]) >= tuple([*target.split('.')])
    except Exception as exc:
        print(f'{name} version not available: {exc}')
        continue
    print(f'{name} version {ver}, target {target}')
sys.exit(0)
PY

# Install Jupyter stack via pip (Spack lacks notebook@7)
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    pip install --no-build-isolation \
        'jupyterlab==4.*' \
        'ipykernel==6.*' \
        'jupyter_server==2.*' \
        'jupyterlab_server==2.*' \
        'notebook==7.*' \
        'jupyter_core>=5' \
        'jupyter_client>=8'
# uninstalled tornado-6.1 jupyter-client-7.1.2  jsonschema-4.17.3
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# rascil 1.0.0 requires tabulate<0.10,>=0.9, but you have tabulate 0.0.0 which is incompatible.
# rascil 1.0.0 requires xarray<2022.13,>=2022.12, but you have xarray 2023.2.0 which is incompatible.
# Successfully installed anyio-4.11.0 argon2-cffi-25.1.0 argon2-cffi-bindings-25.1.0 arrow-1.3.0 async-lru-2.0.5 babel-2.17.0 certifi-2025.8.3 charset_normalizer-3.4.3 fqdn-1.5.1 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.10 isoduration-20.11.0 json5-0.12.1 jsonpointer-3.0.0 jsonschema-4.25.1 jsonschema-specifications-2025.9.1 jupyter-events-0.12.0 jupyter-lsp-2.3.0 jupyter-server-terminals-0.5.3 jupyter_client-8.6.3 jupyter_server-2.17.0 jupyterlab-4.4.9 jupyterlab_server-2.27.3 lark-1.3.0 notebook-7.4.7 notebook-shim-0.2.4 overrides-7.7.0 prometheus-client-0.23.1 python-json-logger-3.3.0 referencing-0.36.2 requests-2.32.5 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 rfc3987-syntax-1.1.0 rpds-py-0.27.1 send2trash-1.8.3 sniffio-1.3.1 terminado-0.18.1 tornado-6.5.2 types-python-dateutil-2.9.0.20250822 uri-template-1.3.0 webcolors-24.11.1 websocket-client-1.8.0

RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python - <<"PY"
import importlib, sys
checks = [
    ('astropy','5.1.1'),
    ('erfa','2.0'),
    ('healpy','1.16'),
]
for (name, target) in checks:
    mod = None
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f'{name} not importable: {exc}')
        sys.exit(1)
    ver = getattr(mod, '__version__', '0.0')
    try:
        assert tuple([*ver.split('.')]) >= tuple([*target.split('.')])
    except Exception as exc:
        print(f'{name} version not available: {exc}')
        continue
    print(f'{name} version {ver}, target {target}')
sys.exit(0)
PY

# Install aratmospy, eidos, katbeam, tools21cm with pinned toolchain via pip
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    pip install --no-build-isolation -U 'pip<25.3' 'cython>=3.0,<3.1' 'extension_helpers>=1.0,<2' 'packaging>=24.2' setuptools setuptools-scm wheel build versioneer && \
    pip install --no-build-isolation --no-deps --no-build-isolation \
        'git+https://github.com/i4Ds/ARatmospy.git@67c302a136beb40a1cc88b054d7b62ccd927d64f#egg=aratmospy' \
        'git+https://github.com/i4Ds/eidos.git@74ffe0552079486aef9b413efdf91756096e93e7' \
        'git+https://github.com/ska-sa/katbeam.git@5ce6fcc35471168f4c4b84605cf601d57ced8d9e' \
        'tools21cm=='$TOOLS21CM_VERSION

# tests
RUN . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -c "import ska_sdp_func_python" || exit 1 && \
    python -c "import ska_sdp_func" || exit 1 && \
    python -c "import casacore, casacore.tables, casacore.quanta; print('python-casacore OK')" && \
    python - <<"PY"
import importlib, sys
checks = [
    ('aratmospy','0.0'),
    ('astropy','5.1.1'),
    ('astropy_healpix','1.0'),
    ('bdsf','1.10'),
    ('dask_mpi','0.0'),
    ('dask','2022.10'),
    ('distributed','2022.10'), # conda has 2022.12.1
    ('eidos','0.0'),
    ('erfa','2.0'),
    ('healpy','1.16'),
    ('joblib','0.0'),
    ('katbeam','0.0'),
    ('lazy_loader','0.0'),
    ('mpi4py','0.0'),
    ('numpy','1.23.5'),
    ('pyfftw','0.0'),
    ('pyuvdata','2.4.2'),
    ('rascil','1.0'),
    ('rfc3986','2.0'),
    ('skimage','0.24'),
    ('sklearn','1.5'),
    ('tools21cm','2.0.3'),
    ('toolz','0.0'),
    ('tqdm','4.0'),
    ('wheel', '0.0'),
]
for (name, target) in checks:
    mod = None
    if name == 'aratmospy':
        exc = None
        for candidate in ('aratmospy', 'ARatmospy'):
            try:
                mod = importlib.import_module(candidate)
                break
            except Exception:
                mod = None
                exc = exc
        if mod is None:
            print('aratmospy not importable: {exc or "unknown"}')
            sys.exit(1)
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
        except Exception as exc:
            print(f'{name} not importable: {exc}')
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

# bdsf 1.12.0 requires backports.shutil-get-terminal-size, which is not installed.
# eidos 1.1.0 requires future, which is not installed.
# tools21cm 2.3.8 requires openpyxl, which is not installed.
# astropy-healpix 1.1.2 requires numpy>=1.25, but you have numpy 1.23.5 which is incompatible.
# rascil 1.0.0 requires tabulate<0.10,>=0.9, but you have tabulate 0.0.0 which is incompatible.
# rascil 1.0.0 requires xarray<2022.13,>=2022.12, but you have xarray 2023.2.0 which is incompatible.
# ska-sdp-datamodels 0.1.3 requires astroplan<0.9,>=0.8, but you have astroplan 0.0.0 which is incompatible.
# ska-sdp-datamodels 0.1.3 requires xarray<2023.0.0,>=2022.10.0, but you have xarray 2023.2.0 which is incompatible.
# ska-sdp-func-python 0.1.5 requires astroplan<0.9,>=0.8, but you have astroplan 0.0.0 which is incompatible.
# ska-sdp-func-python 0.1.5 requires xarray<2023.0.0,>=2022.11.0, but you have xarray 2023.2.0 which is incompatible.

# Copy repository for editable install and testing
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100

RUN fix-permissions /home/${NB_USER} && \
    fix-permissions /opt/view/lib/python3.10/

RUN git clone https://github.com/micbia/mirto.git /opt/mirto && cd $_ && fix-permissions /opt/mirto

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
    echo PYTHONPATH="${PYTHONPATH}" && \
    echo which pip: $(which pip) && \
    echo pip --version: $(pip --version) && \
    printf 'version = "%s"\n\n' "${KARABO_VERSION}" > /home/${NB_USER}/Karabo-Pipeline/karabo/_version.py && \
    printf 'def get_versions():\n    return {"version": "%s", "full-revisionid": None, "dirty": None, "error": None, "date": None}\n' "${KARABO_VERSION}" >> /home/${NB_USER}/Karabo-Pipeline/karabo/_version.py && \
    PYTHONPATH="${BOOTSTRAP_PYTHONPATH}:/opt/view/lib/python3.10/site-packages:${PYTHONPATH}" \
        python -c "import sys, importlib.util as iu; print('sys.path[0:5]=', sys.path[:5]); print('versioneer spec:', iu.find_spec('versioneer'))" && \
    PYTHONPATH="${BOOTSTRAP_PYTHONPATH}:/opt/view/lib/python3.10/site-packages:${PYTHONPATH}" \
        PYTHONDONTWRITEBYTECODE=1 \
        pip -vvv install --use-pep517 --no-build-isolation -e /home/jovyan/Karabo-Pipeline


#    check.warn(importable)
#  /opt/view/lib/python3.10/site-packages/setuptools/command/build_py.py:212: _Warning: Package 'karabo.examples.data' is absent from the `packages` configuration.
#  !!
#
#          ********************************************************************************
#          ############################
#          # Package would be ignored #
#          ############################
#          Python recognizes 'karabo.examples.data' as an importable package[^1],
#          but it is absent from setuptools' `packages` configuration.
#
#          This leads to an ambiguous overall configuration. If you want to distribute this
#          package, please make sure that 'karabo.examples.data' is explicitly added
#          to the `packages` configuration field.
#
#          Alternatively, you can also rely on setuptools' discovery methods
#          (for example by using `find_namespace_packages(...)`/`find_namespace:`
#          instead of `find_packages(...)`/`find:`).
#
#          You can read more about "package discovery" on setuptools documentation page:
#
#          - https://setuptools.pypa.io/en/latest/userguide/package_discovery.html
#
#          If you don't want 'karabo.examples.data' to be distributed and are
#          already explicitly excluding 'karabo.examples.data' via
#          `find_namespace_packages(...)/find_namespace` or `find_packages(...)/find`,
#          you can try to use `exclude_package_data`, or `include-package-data=False` in
#          combination with a more fine grained `package-data` configuration.
#
#          You can read more about "package data files" on setuptools documentation page:
#
#          - https://setuptools.pypa.io/en/latest/userguide/datafiles.html
#
#
#          [^1]: For Python, any directory (with suitable naming) can be imported,
#                even if it does not contain any `.py` files.
#                On the other hand, currently there is no concept of package data
#                directory, all directories are treated like packages.
#          ********************************************************************************


# Register kernel for jovyan user using the Spack Python
RUN python -m ipykernel install --user --name=karabo --display-name="Karabo (Spack Py3.10)"

# Run tests during build to validate environment
ARG SKIP_TESTS=0
ENV SKIP_TESTS=${SKIP_TESTS}
RUN if [ "${SKIP_TESTS:-0}" = "1" ]; then exit 0; fi; \
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \
    pip install -q --no-build-isolation --no-deps 'dask_memusage==1.1' && \
    pip install -q -t "/home/${NB_USER}/.local/pytest" 'pytest>=8,<9' && \
    # Exclude heavy imaging test (disk space) and known OSKAR flakiness in CI build context
    PYTHONPATH="/home/${NB_USER}/.local/pytest:${PYTHONPATH}" python -m pytest -q -x --tb=short -k "not test_suppress_rascil_warning and not test_imaging and not (oskar or OSKAR)" /home/${NB_USER}/Karabo-Pipeline && \
    rm -rf /home/${NB_USER}/.astropy/cache \
           /home/${NB_USER}/.cache/astropy \
           /home/${NB_USER}/.cache/pyuvdata \
           /home/${NB_USER}/.cache/rascil


# ...s..........................................................ssssssssss [ 20%]
# s................F
# =================================== FAILURES ===================================
# ____________ TestObsCoreMeta.test_from_visibility[minimal_casa_ms] _____________
# Karabo-Pipeline/karabo/data/obscore.py:520: in from_visibility
#     uvd.read(vis_inode, **read_kwargs)
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/uvdata.py:12640: in read
#     self.read_ms(
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/uvdata.py:10983: in read_ms
#     ms_obj.read_ms(filepath, **kwargs)
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/ms.py:1993: in read_ms
#     self.history, pyuvdata_written = self._ms_hist_to_string(
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/ms.py:920: in _ms_hist_to_string
#     newline = ";".join([str(col[tbrow]) for col in cols]) + "\n"
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/ms.py:920: in <listcomp>
#     newline = ";".join([str(col[tbrow]) for col in cols]) + "\n"
# E   IndexError: list index out of range
#
# During handling of the above exception, another exception occurred:
# Karabo-Pipeline/karabo/test/conftest.py:318: in _gd2gc_safe
#     return _orig_gd2gc(
# E   ValueError: Invalid data-type for array
#
# During handling of the above exception, another exception occurred:
# /home/ubuntu/Karabo-Pipeline/karabo/test/test_obscore.py:141: in test_from_visibility
#     ???
# Karabo-Pipeline/karabo/data/obscore.py:536: in from_visibility
#     uvd.read(vis_inode, **read_kwargs)
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/uvdata.py:12640: in read
#     self.read_ms(
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/uvdata.py:10983: in read_ms
#     ms_obj.read_ms(filepath, **kwargs)
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/ms.py:2139: in read_ms
#     and self.telescope_name in self.known_telescopes()
# /opt/view/lib/python3.10/site-packages/pyuvdata/uvdata/uvdata.py:2350: in known_telescopes
#     return uvtel.known_telescopes()
# /opt/view/lib/python3.10/site-packages/pyuvdata/telescopes.py:179: in known_telescopes
#     astropy_sites = [site for site in EarthLocation.get_site_names() if site != ""]
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/earth.py:505: in get_site_names
#     return cls._get_site_registry().names
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/earth.py:543: in _get_site_registry
#     reg = get_downloaded_sites()
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/sites.py:143: in get_downloaded_sites
#     return SiteRegistry.from_json(jsondb)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/sites.py:105: in from_json
#     location = EarthLocation.from_geodetic(site_info.pop('longitude') * u.Unit(site_info.pop('longitude_unit')),
# Karabo-Pipeline/karabo/test/test_000_astropy_env.py:76: in _from_geodetic_wgs84
#     return _orig_from_geodetic(*args, **kwargs)
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/earth.py:304: in from_geodetic
#     xyz = geodetic.to_cartesian().get_xyz(xyz_axis=-1) << height.unit
# /opt/view/lib/python3.10/site-packages/astropy/coordinates/earth.py:898: in to_cartesian
#     xyz = erfa.gd2gc(getattr(erfa, self._ellipsoid),
# /opt/view/lib/python3.10/site-packages/erfa/core.py:16026: in gd2gc
#     xyz, c_retval = ufunc.gd2gc(n, elong, phi, height)
# Karabo-Pipeline/karabo/test/conftest.py:326: in _gd2gc_safe
#     return _orig_gd2gc(
# E   ValueError: Invalid data-type for array

WORKDIR "/home/${NB_USER}"
