import numpy as np

from karabo.imaging.image import ImageMosaicker
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles
from karabo.test.util import create_compatible_dirty_image


def test_ImageMosaicker(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty = create_compatible_dirty_image(
        vis,
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )

    dirties = dirty.split_image(N=4, overlap=50)
    mosaicker = ImageMosaicker()
    dirty_mosaic = mosaicker.mosaic(dirties)[0]
    assert dirty.data.shape[2:] == dirty_mosaic.data.shape[2:]
    assert np.linalg.norm(dirty.data[0, 0, :, :] - dirty_mosaic.data[0, 0, :, :]) < 1e-6
