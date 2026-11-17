#!/usr/bin/env bash

set -e

# Docker Desktop on Windows runs Linux containers on a WSL2 kernel.  The WSL
# libraries must be visible to the dynamic linker before Python starts; doing
# this from karabo.__init__ would require replacing the running interpreter.
if [[ "$(uname -r)" == *WSL* ]]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec conda run --no-capture-output -n karabo "$@"
