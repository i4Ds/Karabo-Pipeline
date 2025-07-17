# docker build . -f sp5327demo.Dockerfile --tag=d3vnull0/sp5327demo:latest --push

# ---------------------------
# Builder Stage: Build environment, install dependencies, and generate Spack view
# ---------------------------
FROM spack/ubuntu-jammy:0.23.0 AS builder
# note to replicate some of these steps in your own container, you should first:
# . /opt/spack/share/spack/setup-env.sh

# some packages must be installed by apt in addition to spack.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt-get --no-install-recommends install -y \
    # -> wget: casacore packagse misses this as a build dependency
    #7 446.1 ==> Installing casacore-3.6.1-4e6f2sbww43az7spzgi77inyyr4ewura [192/234]
    #7 457.6 sh: 1: wget: not found
    'wget' \
    # -> needed to install ska-sdp-func later by pip
    'cmake' 'libcfitsio-dev' \
    # -> cmake doesn't like spack installed libcurl, or apt installed either!
    # /opt/view/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)
    # 'libcurl4'
    ;

# Clone the custom Spack repository from GitLab
RUN git clone https://gitlab.com/ska-telescope/sdp/ska-sdp-spack.git /opt/ska-sdp-spack && \
    spack repo add /opt/ska-sdp-spack

# Create a new Spack environment which writes to /opt
RUN --mount=type=cache,target=/opt/buildcache \
    mkdir -p /opt/{software,spack_env,view} && \
    spack env create --dir /opt/spack_env && \
    spack env activate /opt/spack_env && \
    spack mirror add --autopush --unsigned mycache file:///opt/buildcache && \
    spack config add "config:install_tree:root:/opt/software" && \
    spack config add "view:/opt/view" && \
    spack add \
    # arm64 hack: 0.3.28 doesn't work
    'openblas@:0.3.27' \
    # python was 3.9 in Karabo, but 3.10 is needed for py-ska-sdp-func-python@0.5
    'python@3.9' \
    # for montagepy 1.0.1
    # 'montage@6.0' \
    'mpich' \
    # exact version from Conda env (1.10.2) not available in spack
    'py-bdsf@=1.12.0' \
    # this is needed anyway for py-bdsf
    'boost+python+numpy' \
    # 'py-dask-mpi' \
    # exact version from Conda env (2022.12.1) not available in spack
    # 'py-dask@=2023.4.1' \
    # 'py-healpy' \
    # 'py-ipython' \
    # 'py-nbconvert' \
    # 'py-nbformat' \
    # 'py-packaging' \
    # 'py-psutil' \
    # for ARatmospy, tools21cm. fftw defaults +mpi
    # 'py-pyfftw' \
    # 'py-requests' \
    # 'py-rfc3986@2:' \
    # for tools21cm
    # 'py-scikit-image' 'py-scikit-learn' \
    # needed for py-rascil@=1.0.0
    # - ska-sdp-func@ git+https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git@main
    # - xarray<2022.13,>=2022.12 # needs fork of sdp
    # exact version from Conda env (>=2022.11) and pip freeze (2023.2) not
    # available in sdp spack, only 'py-xarray@2023.7'.
    # rascil has extremely strict dependencies
    # 'py-xarray@2022.12:2022' \
    # - dask[diagnostics]<2022.13,>=2022.12
    # - ska-sdp-func-python<0.2,>=0.1
    # on rascil it's @main apparently
    # exact version from Conda env (0.1.4) not available in sdp spack, only 0.5.1
    # 'py-ska-sdp-func-python@0.1:0.1' \
    # - reproject<0.10,>=0.9
    # exact version from Conda env (0.9:0.10.0) conflicts with py-ska-sdp-func-python@0.5.1, needs 0.14:
    # can't install here because it forces py-astropy-healpix@1.0.2 on sdp.
    # 'py-reproject@=0.9' \
    # - scipy<1.10,>=1.9
    'py-scipy@=1.9.3' \
    # 'py-scipy@=1.13.1' \
    # - python-casacore<3.6,>=3.5
    # both 'py-casacore@=3.5.2' and 3.5.1 give checksum error?
    'casacore@=3.5.0 +python' \
    # !!! because of pinned casacore, everybeam and wsclean need to be pinned.
    # --> everybeam >=0.6.2 needs casacore 3.6, but...
    # 'everybeam@=0.6.1' \
    # --> wsclean >= 3.5.1 seems to not like casacore 3.5, and wsclean@3.4 needs everybeam 0.6
    'wsclean@=3.4' \
    # - pandas<1.6,>=1.5
    'py-pandas@=1.5.3' \
    # - numpy<1.24,>=1.23
    'py-numpy@=1.23.5' \
    # - matplotlib<3.7,>=3.6
    'py-matplotlib@=3.6.3' \
    # - h5py<3.8,>=3.7
    # hdf5 defaults +mpi
    'py-h5py@=3.7.0' \
    # - astropy-healpix>=0.6 , only 1.0.2 available in sdp
    # 'py-astropy-healpix@0.6:' \
    # exact version from Conda env (0.1.3) not available in sdp spack, only 'py-ska-sdp-datamodels@0.3.3'
    # 'py-ska-sdp-datamodels@0.1:0.1' \
    # - astroplan<0.9,>=0.8
    # 'py-astroplan@=0.8' # only 0.10.1 in sdp \
    # - distributed<2022.13,>=2022.12
    # exact version from docker pip freeze (2022.12.1) only '@=2023.4.1' in sdp
    # 'py-distributed@=2022.12:2022' \
    # - astropy<5.2,>=5.1
    # 'py-astropy@=5.1' fails at prepare_metadata_for_build_wheel \
    # 'py-ducc0@0.27.0:0.27' not available on sdp spack \
    # 'py-mpi4py@=3.1.6' \
    'py-pip@=22.1.2' \
    'py-setuptools@=59.4.0' \
    'py-wheel@=0.37.1' \
    'py-pybind11@=2.13.5' \
    # - photutils<2.0.0,>=1.5.0 \
    # --> issues with setuptools_scm thinking the version is 0.0.0, installed by pip later.
    # 'py-photutils@1.5.0:2.0.0' \
    # 'py-photutils@=1.5.0' \
    # needed for photutils
    'py-extension-helpers' \
    # to test karabo
    'py-pytest' \
    # karabo can't find numexpr installed by spack, pip
    # 'py-numexpr' \
    && \
    spack install --no-check-signature --fail-fast && \
    # second install is for dependencies only.
    # TODO: fix py-casacore checksum error and merge into above
    spack add 'py-casacore@=3.5.2' && \
    spack concretize --force && \
    spack install --no-check-signature --fail-fast --no-checksum --only dependencies \
    # && \
    # spack gc -y
    ;

# Install remaining packages via pip, because they're extremely difficult for some reason
# install these before setuptools upgrade
RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/spack/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
    'ducc0<0.28.0,>=0.27.0' \
    'numexpr==2.10.2' \
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
    # 'tools21cm' \
    'extension_helpers' \
    'astropy<5.2,>=5.1' \
    'cython>=3.0.0,<3.1.0' \
    'setuptools_scm>=6.2' \
    'setuptools>=61.2' \
    ;
RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/spack/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PHOTUTILS=1.11.0 && \
    python -m pip install --no-build-isolation \
    'git+https://github.com/astropy/photutils.git@1.11.0' \
    ;
RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/spack/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
    python -m pip install --no-build-isolation \
    --index-url=https://artefact.skao.int/repository/pypi-all/simple \
    --extra-index-url=https://pypi.org/simple \
    'rascil==1.0.0' \
    ;
# oskarpy not available on pip
RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/spack/share/spack/setup-env.sh && \
    spack env activate /opt/spack_env && \
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
    export OSKAR_INC_DIR=/opt/software/include && \
    export OSKAR_LIB_DIR=/opt/software/lib && \
    python -m pip install --no-build-isolation \
    '/opt/oskar/python' \
    # 'git+https://github.com/OxfordSKA/OSKAR.git@2.8.3#egg=oskarpy&subdirectory=python' \
    ;


# TODO missing from Karabo Conda environment.yaml
# - python was originall 3.9
# - bluebild =0.1.0 (available from https://github.com/AdhocMan/bipp_spack but only main, and will change soon)
# - dask@2022.12.1
# - montagepy =6.0.0 (not available)
# - ska-gridder-nifty-cuda =0.3.0 (not available)
# - tools21cm =2.0.2 (not available, could be installed via pip but astroml is not available)
# - fftw (Cannot select a single "version" for package "py-pyfftw")
# additional spack packages available for the cuda version:
# - py-dask-cuda
# - cuda-cudart (no cpuonly available)
# - libcufft (no cpuonly available)
# - scipy > 1.10

# Optionally, reduce the size by stripping binaries
# RUN find -L /opt/view/* -type f -exec strip -s {} \;

# ---------------------------
# Final Stage: Create a lean runtime image
# ---------------------------
FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2 AS runtime
USER root

# Install minimal dependencies needed to bootstrap spack env
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
    python3 \
    && apt-get clean

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Copy necessary files from builder
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/view /opt/view
COPY --from=builder /opt/spack_env /opt/spack_env
COPY --from=builder /opt/spack /opt/spack
COPY --from=builder /opt/ska-sdp-spack /opt/ska-sdp-spack

# Setup Spack environment
ENV SPACK_ROOT=/opt/spack \
    PATH=/opt/view/bin:/opt/software/bin:/usr/local/bin:/usr/bin:/bin
RUN . /opt/spack/share/spack/setup-env.sh && \
    spack repo add /opt/ska-sdp-spack && \
    spack env activate /opt/spack_env && \
    echo ". /opt/spack/share/spack/setup-env.sh" >> /etc/profile.d/spack.sh && \
    echo "spack env activate /opt/spack_env" >> /etc/profile.d/spack.sh && \
    . /etc/profile.d/spack.sh

# Copy only the needed Karabo code, not the entire repo
COPY karabo /app/karabo
COPY setup.py /app/
COPY requirements.txt /app/
COPY setup.cfg /app/
COPY pyproject.toml /app/
COPY .git /app/.git
WORKDIR /app

# Install versioneer and other build dependencies before installing the package
RUN . /etc/profile.d/spack.sh && \
    python -m pip install versioneer setuptools wheel pytest && \
    python -m pip install -r requirements.txt && \
    python -m pip install --no-deps /app

# Install jupyter and friends
RUN . /etc/profile.d/spack.sh && \
    python -m pip install notebook ipykernel jupyterlab

# Create a startup script that activates the environment
RUN echo '#!/bin/bash' > /usr/local/bin/entrypoint.sh && \
    echo 'source /opt/spack/share/spack/setup-env.sh' >> /usr/local/bin/entrypoint.sh && \
    echo 'spack env activate /opt/spack_env' >> /usr/local/bin/entrypoint.sh && \
    echo 'exec "$@"' >> /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint to our custom script
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# CMD ["/bin/bash", "-l"]

# TEST ME :
# docker build -f spack.Dockerfile . --tag runtime && docker run --rm -it $_ python -m pytest