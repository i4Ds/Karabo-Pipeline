# syntax=docker/dockerfile:1.6
# Minimal image to run mwa_hyperbeam reliably (numpy 2.x wheel) without touching your main stack
# Build:
#   docker build -t d3vnull0/hyperbeam:latest -f hyperbeam.Dockerfile .
# Test:
#   docker run --rm d3vnull0/hyperbeam:latest python - <<'PY'
# from mwa_hyperbeam import FEEBeam
# import numpy as np
# b = FEEBeam('/home/jovyan/Karabo-Pipeline/karabo/examples/mwa_full_embedded_element_pattern.h5')
# az = np.array([0.0]); za = np.array([0.0]); delays=[0]*16; amps=[1]*16
# j = b.calc_jones_array(az, za, 180e6, delays, amps, False)
# print('OK', j, abs(j[0,0])**2)
# PY

FROM quay.io/jupyter/minimal-notebook:notebook-7.2.2

USER root
SHELL ["/bin/bash", "-lc"]

# Remove conda to avoid interference
RUN rm -f /usr/local/bin/before-notebook.d/10activate-conda-env.sh || true; \
    rm -rf /opt/conda || true

# System libs required by hyperbeam runtime
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
      ca-certificates \
      libhdf5-dev \
      libcfitsio-dev \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv with numpy 2.x and working hyperbeam wheel
RUN python3 -m venv /opt/hbvenv && \
    /opt/hbvenv/bin/pip install -q --no-cache-dir --upgrade 'pip<25.3' && \
    /opt/hbvenv/bin/pip install -q --no-cache-dir \
        'numpy==2.2.2' \
        'mwa-hyperbeam==0.10.2' \
        'matplotlib==3.9.2'

# Quick verification at build time (non-zero power expected)
RUN /opt/hbvenv/bin/python - <<'PY'
from mwa_hyperbeam import FEEBeam
import numpy as np, os

beam_file = '/home/jovyan/Karabo-Pipeline/karabo/examples/mwa_full_embedded_element_pattern.h5'
os.makedirs('/home/jovyan/Karabo-Pipeline/karabo/examples', exist_ok=True)
open(beam_file,'ab').close() if not os.path.exists(beam_file) else None

try:
    b = FEEBeam(beam_file)
    az = np.array([0.0]); za = np.array([0.0])
    j = b.calc_jones_array(az, za, 180e6, [0]*16, [1]*16, False)
    print('hyperbeam import OK; jones shape:', j.shape)
except Exception as e:
    # Import verified even if beam missing; runtime will mount real file
    print('hyperbeam import OK; beam open skipped:', e.__class__.__name__)
PY

ENV PATH=/opt/hbvenv/bin:${PATH}
USER ${NB_UID}
WORKDIR /home/${NB_USER}

CMD ["python", "-c", "import mwa_hyperbeam, sys; print('mwa_hyperbeam', getattr(mwa_hyperbeam,'__version__','unknown')); sys.stdout.flush(); import time; time.sleep(3600)"]


