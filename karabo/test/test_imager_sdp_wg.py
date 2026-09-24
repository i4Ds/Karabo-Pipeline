"""GPU gridder path of the SDP imaging backend (context="wg").

ska-sdp-func-python routes context="wg" to ska-sdp-func's GridderUvwEsFft, the GPU
port of the nifty/WAGG w-stacking gridder. It needs the CUDA kernels in ska-sdp-func
and cupy at runtime; without cupy invert_visibility(context="wg") raises
"cupy is not installed. Cannot run invert_wg".
"""

import os
from pathlib import Path

import numpy as np
import pytest
from casacore.tables import table
from numpy.typing import NDArray

from karabo.imaging.imager_factory import ImagingBackend, SdpImagerConfig, get_imager
from karabo.imaging.imager_interface import ImageSpec
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import RUN_GPU_TESTS


def test_sdp_wg_context_dependencies_installed() -> None:
    """Runs on CPU-only CI too: catches a missing runtime dependency of context="wg".

    ska_sdp_func_python.imaging.wg sets these names to None when the package is
    absent and only fails later, at imaging time, on a machine with a GPU.
    """
    from ska_sdp_func_python.imaging import wg

    assert wg.cupy is not None, "cupy is not installed: SDP context='wg' cannot run"
    assert wg.GridderUvwEsFft is not None, "ska-sdp-func gridder is not importable"


def _image_spec_from_ms(
    ms_path: str | Path, npix: int, pixels_per_beam: float
) -> ImageSpec:
    """Cell size from the finest resolution in the MS, lambda_min / max |(u, v)|."""
    ms_path = os.fspath(ms_path)
    with table(ms_path, ack=False) as t:
        uvw = t.getcol("UVW")  # metres
    with table(f"{ms_path}/SPECTRAL_WINDOW", ack=False) as spw:
        freq_max_hz = spw.getcol("CHAN_FREQ").max()
    with table(f"{ms_path}/FIELD", ack=False) as field:
        ra_rad, dec_rad = field.getcol("PHASE_DIR")[0][0]
    baseline_max_m = np.sqrt((uvw[:, :2] ** 2).sum(axis=1)).max()
    resolution_rad = (299792458.0 / freq_max_hz) / baseline_max_m
    return ImageSpec(
        npix=npix,
        cellsize_arcsec=float(np.degrees(resolution_rad) * 3600.0 / pixels_per_beam),
        phase_centre_deg=(
            float(np.degrees(ra_rad) % 360.0),
            float(np.degrees(dec_rad)),
        ),
    )


@pytest.mark.skipif(not RUN_GPU_TESTS, reason="GPU tests are disabled")
def test_sdp_wg_context_matches_ng(minimal_casa_ms: Visibility) -> None:
    """The GPU gridder ("wg") must reproduce the CPU nifty gridder ("ng") result.

    Like the other GPU tests this only runs with RUN_GPU_TESTS=true, i.e. on a
    machine with a CUDA GPU (CI runners have none); the dependency test above is
    the part that runs everywhere.
    """
    spec = _image_spec_from_ms(minimal_casa_ms.path, npix=256, pixels_per_beam=3.0)
    images: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for context in ("ng", "wg"):
        imager = get_imager(ImagingBackend.SDP, config=SdpImagerConfig(context=context))
        dirty, psf = imager.invert(minimal_casa_ms, spec)
        images[context] = (dirty.get_squeezed_data(), psf.get_squeezed_data())

    for kind, (ng, wg) in zip(("dirty", "psf"), zip(*images.values())):
        assert np.isfinite(wg).all()
        scale = np.max(np.abs(ng))
        assert scale > 0.0, f"{kind}: reference ({'ng'}) image is all zeros"
        diff = np.max(np.abs(ng - wg))
        print(
            f"{kind}: max|ng|={scale:.4g} max|ng-wg|={diff:.3g} rel={diff / scale:.2e}"
        )
        assert diff <= 1e-8 * scale
