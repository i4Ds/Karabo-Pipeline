import math
import os

import numpy as np
import pytest

from karabo.imaging.backends.sdp_backend import SdpImager, SdpImagerConfig
from karabo.imaging.imager_factory import (
    ImagingBackend,
    get_imager,
    parse_imaging_backend,
)
from karabo.imaging.imager_interface import ImageSpec
from karabo.imaging.util import guess_beam_parameters
from karabo.simulation.visibility import Visibility


@pytest.mark.usefixtures("minimal_casa_ms")
def test_sdp_imager_invert_and_restore(minimal_casa_ms: Visibility) -> None:
    imager = get_imager(ImagingBackend.SDP)
    spec = ImageSpec(
        npix=256,
        cellsize_arcsec=math.degrees(5e-5) * 3600.0,
        phase_centre_deg=(0.0, 0.0),
    )

    dirty_image, psf_image = imager.invert(minimal_casa_ms, spec)

    assert os.path.exists(dirty_image.path)
    assert os.path.exists(psf_image.path)
    assert dirty_image.data.shape == psf_image.data.shape
    assert dirty_image.data.size > 0
    assert psf_image.data.size > 0
    assert not np.allclose(
        dirty_image.get_squeezed_data(), psf_image.get_squeezed_data()
    )

    psf_data = psf_image.get_squeezed_data()
    psf_peak = np.unravel_index(np.nanargmax(psf_data), psf_data.shape)
    image_centre = tuple(size // 2 for size in psf_data.shape)
    assert (
        max(abs(actual - expected) for actual, expected in zip(psf_peak, image_centre))
        <= 1
    )

    beam = guess_beam_parameters(psf_image)
    assert np.isfinite([beam["bmaj"], beam["bmin"], beam["bpa"]]).all()
    assert beam["bmaj"] > 0.0
    assert beam["bmin"] > 0.0

    restored = imager.restore(dirty_image, psf_image)
    assert os.path.exists(restored.path)
    assert restored.data.shape == dirty_image.data.shape
    assert restored.data.size > 0
    # model/residual artefacts are exported for inspection
    assert hasattr(imager, "last_model_image")
    assert hasattr(imager, "last_residual_image")
    assert os.path.exists(imager.last_model_image.path)
    assert os.path.exists(imager.last_residual_image.path)


@pytest.mark.parametrize("algorithm", ["hogbom-complex", "msclean", "mmclean"])
def test_sdp_imager_config_rejects_unsupported_clean_algorithm(
    algorithm: str,
) -> None:
    with pytest.raises(NotImplementedError, match="only clean_algorithm='hogbom'"):
        SdpImagerConfig(clean_algorithm=algorithm)


def test_sdp_imager_restore_rejects_mutated_clean_algorithm() -> None:
    imager = SdpImager()
    imager.config.clean_algorithm = "msclean"
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        imager.restore(None, None)  # type: ignore[arg-type]


def test_removed_rascil_backend_has_migration_error() -> None:
    with pytest.raises(
        ValueError,
        match="RASCIL imaging backend has been removed.*'sdp' or 'wsclean'",
    ):
        parse_imaging_backend("rascil")
