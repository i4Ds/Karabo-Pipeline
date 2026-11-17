#!/bin/bash

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

if [[ -z "${KARABO_VERSION:-}" ]]; then
    echo "KARABO_VERSION must be set for a Conda build" >&2
    exit 1
fi

# Keep the version exposed by the Python package identical to the version in
# conda-meta. Versioneer cannot determine it after the installed package has
# been detached from its Git checkout.
"$PYTHON" -c 'import os; from pathlib import Path; Path("karabo/_build_version.py").write_text("__version__ = " + repr(os.environ["KARABO_VERSION"]) + "\n")'

$PYTHON -m pip install --no-deps .
