#!/usr/bin/env bash
set -euo pipefail

# Config
SPACK_ROOT=/opt/spack
ENV_DIR="$HOME/spack_env_karabo"
VIEW_DIR="$HOME/spack-view"
INSTALL_DIR="$HOME/spack-software"
OSKAR_VER=2.8.3
OSKAR_PREFIX="$INSTALL_DIR/oskar"

# 1) System packages
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  build-essential gfortran cmake pkg-config git curl wget ca-certificates \
  zstd patchelf flex bison autoconf automake libtool nasm pigz tar xz-utils \
  libcfitsio-dev libfftw3-dev libhdf5-dev libcurl4-openssl-dev \
  libreadline-dev libexpat1-dev libxml2-dev libpng-dev libfreetype6-dev \
  libjpeg-turbo8-dev libgsl-dev libyaml-dev libopenblas-dev \
  mpich libmpich-dev libfabric-dev libnghttp2-dev libedit-dev \
  libssh2-1-dev libidn2-dev libunistring-dev wcslib-dev \
  casacore-dev
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*

# 2) Spack (fresh install)
if [ ! -d "$SPACK_ROOT" ]; then
  sudo git clone --depth=2 --branch releases/v0.23 https://github.com/spack/spack.git "$SPACK_ROOT"
  sudo chown -R "$USER":"$USER" "$SPACK_ROOT"
fi
. "$SPACK_ROOT/share/spack/setup-env.sh"

# 3) Spack env + config + core specs
if [ ! -d "$ENV_DIR" ]; then
  spack env create -d "$ENV_DIR"
fi
spack env activate -d "$ENV_DIR"
spack config add 'concretizer:unify:when_possible'
spack config add "view:$VIEW_DIR"
spack config add "config:install_tree:root:$INSTALL_DIR"

# Core specs: Python 3.10 and Boost with Python runtime
spack add python@3.10
spack add boost+python+numpy
spack concretize --force
spack install -y --fail-fast
spack env view regenerate

PYBIN="$VIEW_DIR/bin/python3"

# 4) Base Python tooling (avoid pip>=25.3 for legacy builds)
"$PYBIN" -m pip install -q --no-input --upgrade 'pip<25.3' 'setuptools<81' wheel build versioneer

# 5) Scientific/runtime pins known-good with Karabo/OSKAR/SKA stack
"$PYBIN" -m pip install -q --no-input \
  'numpy==1.23.5' 'scipy==1.9.3' 'pandas==1.5.3' 'xarray==2022.12.0' 'h5py==3.7' \
  'dask==2022.12' 'distributed==2022.12' 'dask_memusage>=1.1' 'tabulate>=0.9' \
  'astropy==5.1' 'matplotlib==3.6.*' 'astropy-healpix==1.0.0' \
  'pyuvdata==2.4.2' 'healpy==1.16.2' 'rfc3986>=2.0.0' tools21cm requests

# 6) Notebooks + misc
"$PYBIN" -m pip install -q --no-input nbformat nbclient 'nbconvert==7.*' nest-asyncio jupyter_client

# 7) SKA pins (install via SKAO index + VCS pins)
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPLAN=0.8
"$PYBIN" -m pip install -q --no-build-isolation --no-deps \
  'git+https://github.com/astropy/astroplan.git@v0.8'

"$PYBIN" -m pip install -q --no-input \
  --index-url=https://artefact.skao.int/repository/pypi-all/simple \
  --extra-index-url=https://pypi.org/simple \
  'ska-sdp-datamodels==0.1.3'

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PHOTUTILS=1.11.0
"$PYBIN" -m pip install -q --no-build-isolation --no-deps \
  'git+https://github.com/astropy/photutils.git@1.11.0'

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_REPROJECT=0.9.1
"$PYBIN" -m pip install -q --no-build-isolation --no-deps \
  'git+https://github.com/astropy/reproject.git@v0.9.1'

"$PYBIN" -m pip install -q --no-build-isolation --no-deps \
  --index-url=https://artefact.skao.int/repository/pypi-all/simple \
  --extra-index-url=https://pypi.org/simple \
  'ska-sdp-func-python==0.1.5' \
  'git+https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git@08eb17cf9f4d63320dd0618032ddabc6760188c9'

# 8) MPI Python stack
MPICC=/usr/bin/mpicc.mpich "$PYBIN" -m pip install -q --no-input --no-binary=:all: mpi4py
"$PYBIN" -m pip install -q --no-input dask_mpi

# 9) python-casacore (build against apt casacore)
export CASACORE_ROOT=/usr
export CMAKE_PREFIX_PATH="/usr:/usr/local"
export CPATH="/usr/include:${CPATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
"$PYBIN" -m pip install -q --no-binary=:all: --no-build-isolation 'python-casacore==3.5.0'

# 10) PyBDSF (with legacy build toolchain pins)
"$PYBIN" -m pip install -q --no-input 'Cython==0.29.36' 'setuptools_scm>=8' scikit-build
SETUPTOOLS_SCM_PRETEND_VERSION=1.12.0 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BDSF=1.12.0 \
  "$PYBIN" -m pip install -q --no-build-isolation 'git+https://github.com/lofar-astron/PyBDSF.git@v1.12.0' || \
  "$PYBIN" -m pip install -q --no-build-isolation --only-binary=:all: 'bdsf==1.12.0'

# 11) OSKAR build + Python bindings
if [ ! -d /opt/oskar ]; then
  sudo git clone https://github.com/OxfordSKA/OSKAR.git /opt/oskar
  sudo chown -R "$USER":"$USER" /opt/oskar
fi
pushd /opt/oskar >/dev/null
git fetch --tags -q || true
git checkout "$OSKAR_VER"
rm -rf build && mkdir build && cd build
# Link against Spack view + system HDF5/cfitsio
export CMAKE_PREFIX_PATH="$VIEW_DIR:/usr"
cmake -DCMAKE_INSTALL_PREFIX="$OSKAR_PREFIX" -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"
make install
popd >/dev/null

# OSKAR Python bindings (no isolation; uses env above)
export OSKAR_INC_DIR="$OSKAR_PREFIX/include"
export OSKAR_LIB_DIR="$OSKAR_PREFIX/lib"
export LD_LIBRARY_PATH="$OSKAR_LIB_DIR:/usr/lib/x86_64-linux-gnu/hdf5/serial:$LD_LIBRARY_PATH"
"$PYBIN" -m pip install -q --no-build-isolation '/opt/oskar/python'

# 12) seqfile (rascil dep)
"$PYBIN" -m pip install -q --no-input 'seqfile>=0.2,<0.3'

# 13) Runtime env helper
BOOST_LIB=$(spack -e "$ENV_DIR" location -i boost)/lib
cat > "$HOME/.karabo_env" <<EOF
# Karabo runtime env
. "$SPACK_ROOT/share/spack/setup-env.sh"
spack env activate -d "$ENV_DIR" >/dev/null 2>&1 || true
export PATH="$OSKAR_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="$OSKAR_PREFIX/lib:$BOOST_LIB:/usr/lib/x86_64-linux-gnu/hdf5/serial:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
EOF

echo
echo "Done. To use the environment in new shells:"
echo "  source \$HOME/.karabo_env"
echo
echo "Quick check:"
. "$HOME/.karabo_env"
"$PYBIN" - <<'PY'
import os, sys
mods = ["numpy","scipy","astropy","xarray","casacore","oskar","mpi4py"]
ok = True
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        ok = False
        print(f"[FAIL] import {m}: {e}")
    else:
        print(f"[OK]   import {m}")
sys.exit(0 if ok else 1)
PY