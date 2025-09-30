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

# Version pins aligned with sp5505
ARG ASTROPLAN_VERSION=0.10.1
# 0.8 may have worked at some point
# conda uses 0.10.1
ARG ASTROPY_HEALPIX_VERSION=1.1.2
# 1.0.0 was installed at some point
# conda uses 1.1.2
ARG BDSF_VERSION=1.10.2
# 1.12.0 worked at some point
# conda uses 1.10.2
ARG DASK_VERSION=2022.12.1
ARG DUCC_VERSION=0.27
ARG PYERFA_VERSION=2.0.1.5
# 2.0.0 worked at some point
# 2.0.0.1 needed patches to work with astropy 5.1
ARG PHOTUTILS_VERSION=1.11.0
# 1.11.0 worked at some point
# conda uses 1.8.0
ARG REPROJECT_VERSION=0.9.1
# 0.9.1 worked at some point
# conda uses 0.14.1
ARG SDP_DATAMODELS_VERSION=0.1.3
ARG SDP_FUNC_VERSION=1.1.7
# 0.1.5 worked at some point
# conda uses 0.0.6
# only 1.1.7 in sdp-spack
ARG SDP_FUNC_PYTHON_VERSION=0.1.4
# conda uses 0.1.4
ARG RASCIL_VERSION=1.0.0
# conda uses 1.0.0

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

ARG TOOLS21CM_VERSION=2.0.3
# tools21cm needed by rascil ska-sdp-func-python aratmospy bdsf tools21cm gwcs photutils ska-sdp-datamodels eidos bluebild
# tools21cm 2.0.3 installed by conda
# tools21cm 2.8.3 is available

ARG TABULATE_VERSION=0.9.0
# conda has 0.9.0
ARG NUMEXPR_VERSION=2.10.2
# conda has 2.10.2
ARG BOTTLENECK_VERSION=1.5.0
# conda has 1.5.0
ARG SEQFILE_VERSION=0.2.0
# conda has 0.2.0
ARG OPENBLAS_VERSION=0.3.27
# 0.3.27 works on arm64
# conda has 0.3.30
# 0.3.28 is the latest supported in builtin
ARG BOOST_VERSION=1.82.0
# 1.86.0 worked at some point
# conda has 1.82.0

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
    spack config add "concretizer:unify:when_possible"; \
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

        "py-packaging" \
        'py-pandas@'$PANDAS_VERSION \

        'py-astroplan@'$ASTROPLAN_VERSION \
        'py-astropy@'$ASTROPY_VERSION \
        'py-ducc@'$DUCC_VERSION \

        "py-scikit-build" \
        'py-scikit-image' \
        'py-scikit-learn' \
        'py-photutils@'$PHOTUTILS_VERSION \

        'py-scipy@'$SCIPY_VERSION \

        'hdf5@'$HDF5_VERSION'+hl~mpi' \
        'py-h5py@'$H5PY_VERSION'~mpi' \
        'py-ska-sdp-datamodels@'$SDP_DATAMODELS_VERSION \

        'py-ska-sdp-func@'$SDP_FUNC_VERSION \
        'py-ska-sdp-func-python@'$SDP_FUNC_PYTHON_VERSION \
    && \
    spack --verbose concretize --force && \
    ac_cv_lib_curl_curl_easy_init=no spack install --no-check-signature --no-checksum --fail-fast --reuse || ( \
        for f in /tmp/root/spack-stage/spack-stage-py-ska-sdp*/spack-build-out.txt; do \
            echo "=== $f ==="; \
            cat $f; \
        done; \
        exit 1; \
    ) && \
    spack env view regenerate && \
    spack test run 'py-astroplan' && \
    spack test run 'py-astropy' && \
    spack test run 'py-photutils' && \
    spack test run 'py-scipy' && \
    spack test run 'py-ska-sdp-datamodels' && \
    spack test run 'py-ska-sdp-func' && \
    spack test run 'py-ska-sdp-func-python'
