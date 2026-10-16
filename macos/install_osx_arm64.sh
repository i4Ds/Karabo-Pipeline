#!/usr/bin/env bash
# Native Apple Silicon install of Karabo: no CUDA, no Rosetta.
#
#   bash macos/install_osx_arm64.sh            # env name: karabo-osx
#   KARABO_ENV=myenv bash macos/install_osx_arm64.sh
#
# Steps:
#   1. conda env from environment-osx-arm64.yaml
#   2. WSClean (arm64 bottle from the mwatelescope Homebrew tap)
#   3. OSKAR + oskarpy built from source into the env (OpenCL: Apple GPU)
#   4. pip packages without an osx-arm64 conda build, with --no-deps
#   5. Karabo itself (editable)
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This script is for macOS on Apple Silicon (arm64) only." >&2
    exit 1
fi
if [[ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" == "1" ]]; then
    echo "This shell runs under Rosetta. Open a native arm64 shell." >&2
    exit 1
fi

ENV_NAME="${KARABO_ENV:-karabo-osx}"
OSKAR_VERSION="${OSKAR_VERSION:-2.12.2}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${KARABO_BUILD_DIR:-${REPO_DIR}/.build-osx-arm64}"
SKA_INDEX="https://artefact.skao.int/repository/pypi-internal/simple"

CONDA="$(command -v mamba || command -v conda)"

echo "==> conda env ${ENV_NAME}"
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA" env update -n "$ENV_NAME" -f "${REPO_DIR}/environment-osx-arm64.yaml"
else
    "$CONDA" env create -n "$ENV_NAME" -f "${REPO_DIR}/environment-osx-arm64.yaml"
fi
PREFIX="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
export PATH="${PREFIX}/bin:${PATH}"
PIP=("${PREFIX}/bin/python" -m pip install --no-deps --no-build-isolation)

echo "==> WSClean"
if ! command -v wsclean >/dev/null; then
    brew install mwatelescope/tap/wsclean
fi
wsclean -version | head -2

echo "==> OSKAR ${OSKAR_VERSION}"
mkdir -p "$BUILD_DIR"
if [[ ! -d "${BUILD_DIR}/OSKAR" ]]; then
    git clone --depth 1 --branch "$OSKAR_VERSION" \
        https://github.com/OxfordSKA/OSKAR.git "${BUILD_DIR}/OSKAR"
fi
# Same patch as the i4ds conda build (i4Ds/Karabo-Feedstock, oskar/0001-*).
PATCH="${REPO_DIR}/macos/patches/oskar-0001-fix-optional-phase-convention-tag.patch"
if git -C "${BUILD_DIR}/OSKAR" apply --check "$PATCH" 2>/dev/null; then
    git -C "${BUILD_DIR}/OSKAR" apply "$PATCH"
fi
cmake -S "${BUILD_DIR}/OSKAR" -B "${BUILD_DIR}/OSKAR/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_INSTALL_RPATH="${PREFIX}/lib" \
    -DCASACORE_INC_DIR="${PREFIX}/include" \
    -DCASACORE_LIB_DIR="${PREFIX}/lib" \
    -DFIND_CUDA=OFF -DFIND_OPENCL=ON -DBUILD_TESTING=OFF
cmake --build "${BUILD_DIR}/OSKAR/build"
cmake --install "${BUILD_DIR}/OSKAR/build"
(cd "${BUILD_DIR}/OSKAR/python" &&
    OSKAR_INC_DIR="${PREFIX}/include" OSKAR_LIB_DIR="${PREFIX}/lib" \
        "${PIP[@]}" .)

echo "==> ska-sdp-func ${SKA_SDP_FUNC_VERSION:=1.0.0} (CPU only)"
if [[ ! -d "${BUILD_DIR}/ska-sdp-func" ]]; then
    git clone --depth 1 --branch "$SKA_SDP_FUNC_VERSION" \
        https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git \
        "${BUILD_DIR}/ska-sdp-func"
fi
CMAKE_ARGS="-DFIND_CUDA=OFF" "${PIP[@]}" "${BUILD_DIR}/ska-sdp-func"

echo "==> pip-only packages"
"${PIP[@]}" "bdsf==1.14.1" "MontagePy==2.3.0" "tools21cm==2.4.3"
"${PIP[@]}" --extra-index-url "$SKA_INDEX" \
    "ska-sdp-datamodels==1.4.1" "ska-sdp-func-python==1.0.0"

echo "==> Karabo"
"${PIP[@]}" -e "$REPO_DIR"

echo "==> smoke test"
(cd "$BUILD_DIR" && python -c "
import platform, oskar, bdsf, karabo
from karabo.simulation.sample_simulation import run_sample_simulation
print('karabo', karabo.__version__, '| oskar', oskar.__version__,
      '|', platform.machine())
")
echo "Done. Activate with: conda activate ${ENV_NAME}"
