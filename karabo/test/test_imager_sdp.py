import math
import os

import numpy as np
import pytest

from karabo.imaging.imager_factory import ImagingBackend, get_imager
from karabo.imaging.imager_interface import ImageSpec
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

    restored = imager.restore(dirty_image, psf_image)
    assert os.path.exists(restored.path)
    assert restored.data.shape == dirty_image.data.shape
    assert restored.data.size > 0
    # model/residual artefacts are exported for inspection
    assert hasattr(imager, "last_model_image")
    assert hasattr(imager, "last_residual_image")
    assert os.path.exists(imager.last_model_image.path)
    assert os.path.exists(imager.last_residual_image.path)
