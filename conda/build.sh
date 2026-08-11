#!/bin/bash

export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

$PYTHON -m pip install --no-deps .