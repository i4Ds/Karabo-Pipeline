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

# Create Spack environment, enable view at /opt/view, add packages, install
# first install only the packages that fail when installing with -j$(nproc)
# spack env deactivate || true; \
# spack env rm -y /opt/spack_env || true; \
# spack mirror rm -y mycache || true; \
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
        "cfitsio@:4" \
    && \
    spack concretize --force && \
    export ac_cv_lib_curl_curl_easy_init=no PKG_CONFIG_PATH=""; \
    spack install --no-check-signature --fail-fast \
    || cat /tmp/root/spack-stage/spack-stage-*/spack-build-out.txt && \
    spack env view regenerate

# these are safe with -j$(nproc)
RUN --mount=type=cache,target=/opt/buildcache,id=spack-binary-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-source-cache,id=spack-source-cache,sharing=locked \
    --mount=type=cache,target=/opt/spack-misc-cache,id=spack-misc-cache,sharing=locked \
    . ${SPACK_ROOT}/share/spack/setup-env.sh; \
    spack env activate /opt/spack_env; \
    spack add \
      "boost+python+numpy" \
      "casacore@=3.5.0 +python" \
      "mpich" \
      "openblas@:0.3.27" \
      "py-bdsf@=1.12.0" \
      "py-extension-helpers" \
      "py-h5py@=3.7.0" \
      "py-matplotlib@=3.6.3" \
      "py-numpy@:2" \
      "py-pandas@=1.5.3" \
      "py-pip@=22.1.2" \
      "py-pybind11@=2.13.5" \
      "py-pytest" \
      "py-scipy@=1.9.3" \
      "py-setuptools@=59.4.0" \
      "py-wheel@=0.37.1" \
      "python@3.10" \
      "wsclean@=3.4" \
    ; \
    spack concretize --force && \
    spack install --no-check-signature --fail-fast && \
    spack env view regenerate

# Make Spack view default in system paths and shells
RUN printf "/opt/view/lib\n/opt/view/lib64\n" > /etc/ld.so.conf.d/spack-view.conf && ldconfig && \
    echo ". ${SPACK_ROOT}/share/spack/setup-env.sh 2>/dev/null || true" > /etc/profile.d/spack.sh && \
    echo "spack env activate /opt/spack_env 2>/dev/null || true" >> /etc/profile.d/spack.sh && \
    mkdir -p /opt/etc && \
    echo ". /etc/profile.d/spack.sh" > /opt/etc/spack_env && \
    chmod 644 /opt/etc/spack_env

# Prefer Spack view in PATH, and ensure batch shells source spack env
ENV PATH="/opt/view/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/view/lib:/opt/view/lib64:${LD_LIBRARY_PATH}" \
    BASH_ENV=/opt/etc/spack_env \
    PYTHONNOUSERSITE=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Pre-install build-time requirements needed without build isolation
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
        'cython>=3.0.0,<3.1.0'

# Install ipykernel, pytest, and karabo-pipeline into Spack Python (as root)
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
        'ipykernel' \
        'pytest' \
        'ducc0<0.28.0,>=0.27.0' \
        'numexpr==2.10.2' \
        'xarray' \
        'requests' \
        'tqdm' \
        'healpy' \
        'git+https://github.com/i4Ds/eidos.git@74ffe0552079486aef9b413efdf91756096e93e7' \
        'git+https://github.com/ska-sa/katbeam.git@5ce6fcc35471168f4c4b84605cf601d57ced8d9e' \
        'dask_mpi' \
        'rfc3986>=2.0.0' \
        'git+https://github.com/i4Ds/ARatmospy.git@67c302a136beb40a1cc88b054d7b62ccd927d64f' \
        'nbconvert' \
        'nbformat' \
        'nest_asyncio' \
        'tools21cm' \
        'extension_helpers' \
        'versioneer' \
        'astropy<5.2,>=5.1' \
        'setuptools_scm>=6.2' \
        'setuptools>=61.2'


# installs: PyYAML-6.0.2 aratmospy-1.0.0 astropy-5.1.1 asttokens-3.0.0 attrs-25.3.0 beautifulsoup4-4.13.5 bleach-6.2.0 certifi-2025.8.3 charset_normalizer-3.4.3 comm-0.2.3 dask_mpi-2022.4.0 debugpy-1.8.16 decorator-5.2.1 defusedxml-0.7.1 ducc0-0.27.0 eidos-1.1.0 et-xmlfile-2.0.0 executing-2.2.1 extension_helpers-1.4.0 fastjsonschema-2.21.2 future-1.0.0 healpy-1.18.1 idna-3.10 imageio-2.37.0 ipykernel-6.30.1 ipython-8.37.0 jedi-0.19.2 jinja2-3.1.6 joblib-1.5.2 jsonschema-4.25.1 jsonschema-specifications-2025.9.1 jupyter-client-8.6.3 jupyter-core-5.8.1 jupyterlab-pygments-0.3.0 katbeam-0.2.dev36+head.5ce6fcc lazy-loader-0.4 markupsafe-3.0.2 matplotlib-inline-0.1.7 mistune-3.1.4 nbclient-0.10.2 nbconvert-7.16.6 nbformat-5.10.4 nest_asyncio-1.6.0 networkx-3.4.2 numexpr-2.10.2 openpyxl-3.1.5 pandas-2.3.2 pandocfilters-1.5.1 parso-0.8.5 pexpect-4.9.0 platformdirs-4.4.0 prompt_toolkit-3.0.52 psutil-7.0.0 ptyprocess-0.7.0 pure-eval-0.2.3 pyerfa-2.0.1.5 pyfftw-0.15.0 pygments-2.19.2 pyzmq-27.1.0 referencing-0.36.2 requests-2.32.5 rfc3986-2.0.0 rpds-py-0.27.1 scikit-image-0.25.2 scikit-learn-1.7.2 scipy-1.15.3 setuptools-80.9.0 setuptools_scm-9.2.0 soupsieve-2.8 stack_data-0.6.3 threadpoolctl-3.6.0 tifffile-2025.5.10 tinycss2-1.4.0 tools21cm-2.3.8 tornado-6.5.2 tqdm-4.67.1 traitlets-5.14.3 typing_extensions-4.15.0 tzdata-2025.2 urllib3-2.5.0 versioneer-0.29 wcwidth-0.2.13 webencodings-0.5.1 xarray-2025.6.1
# replaces:
#   SciPy 1.9.3 -> 1.15.3
#   setuptools 59.4.0 -> 80.9.0
#   astropy 4.0.1.post1 -> 5.1.1
# xarray reinstalls:
#   pandas 1.5.3 -> 2.3.2
#


# ska-sdp-datamodels 0.1.0 - 3 depends on astroplan<0.9 and >=0.8

# astropy-photutils
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PHOTUTILS=1.11.0
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
    'git+https://github.com/astropy/photutils.git@1.11.0'
# rascil (optional; disabled by default due to dependency conflicts)
# https://artefact.skao.int/service/rest/repository/browse/pypi-all/rascil/
# this reinstalls pip 22.1.2 -> 25.2, wheel 0.37.1 -> 0.45.1
ARG SKIP_RASCIL=0
ENV SKIP_RASCIL=${SKIP_RASCIL}
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    if [ "${SKIP_RASCIL:-1}" = "1" ]; then \
      echo "Skipping rascil install"; \
    else \
      python -m pip install --upgrade pip wheel setuptools && \
      python -m pip install --no-build-isolation \
        --index-url=https://artefact.skao.int/repository/pypi-all/simple \
        --extra-index-url=https://pypi.org/simple \
        'rascil==1.0.0' || exit 1; \
    fi
# oskarpy not available on pip
ARG SKIP_OSKARPY=0
ENV SKIP_OSKARPY=${SKIP_OSKARPY}
ENV OSKAR_INC_DIR=/opt/software/include \
    OSKAR_LIB_DIR=/opt/software/lib
RUN --mount=type=cache,target=/root/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    if [ "${SKIP_OSKARPY:-1}" = "1" ]; then \
        echo "Skipping oskarpy install"; \
    else \
        git clone 'https://github.com/OxfordSKA/OSKAR.git' /opt/oskar && \
        cd /opt/oskar/ && \
        # karabo uses 2.8.3 but 2.10.0 works on arm64
        git checkout '2.10.0' && \
        mkdir build && \
        cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/opt/software -DCMAKE_BUILD_TYPE=Release .. && \
        # too many threads crashes arm64
        make -j1 && \
        make install && \
        python -m pip install --no-build-isolation \
        '/opt/oskar/python' || exit 1; \
    fi
    # 'git+https://github.com/OxfordSKA/OSKAR.git@2.8.3#egg=oskarpy&subdirectory=python'

# Copy repository for editable install and testing
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100
COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/Karabo-Pipeline

RUN fix-permissions /home/${NB_USER} && \
    fix-permissions /home/${NB_USER}/Karabo-Pipeline
USER ${NB_UID}
# Set explicit version for Karabo-Pipeline when building without VCS metadata
ARG KARABO_VERSION=0.34.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_KARABO_PIPELINE=${KARABO_VERSION} \
    SETUPTOOLS_SCM_PRETEND_VERSION=${KARABO_VERSION} \
    VERSIONEER_OVERRIDE=${KARABO_VERSION}

RUN --mount=type=cache,target=/home/${NB_USER}/.cache/pip \
    . ${SPACK_ROOT}/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    printf 'version = "%s"\n\n' "${KARABO_VERSION}" > /home/${NB_USER}/Karabo-Pipeline/karabo/_version.py && \
    printf 'def get_versions():\n    return {"version": "%s", "full-revisionid": None, "dirty": None, "error": None, "date": None}\n' "${KARABO_VERSION}" >> /home/${NB_USER}/Karabo-Pipeline/karabo/_version.py && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_KARABO_PIPELINE=${KARABO_VERSION} SETUPTOOLS_SCM_PRETEND_VERSION=${KARABO_VERSION} VERSIONEER_OVERRIDE=${KARABO_VERSION} && \
    python -m pip install --no-build-isolation -e /home/jovyan/Karabo-Pipeline

# Register kernel for jovyan user using the Spack Python
RUN python -m ipykernel install --user --name=karabo --display-name="Karabo (Spack Py3.10)"

# Run tests during build to validate environment
ARG SKIP_TESTS=0
ENV SKIP_TESTS=${SKIP_TESTS}
RUN if [ "${SKIP_TESTS:-0}" = "1" ]; then exit 0; fi; pytest -q -x --tb=short /home/${NB_USER}/Karabo-Pipeline

WORKDIR "/home/${NB_USER}"
