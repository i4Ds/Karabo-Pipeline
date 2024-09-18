import os

from karabo.imaging.imager_oskar import OskarDirtyImager, OskarDirtyImagerConfig
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles
from karabo.util.file_handler import FileHandler


def test_dirty_image(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty_imager = OskarDirtyImager(
        OskarDirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(vis)

    assert dirty_image.data.ndim == 4


def test_dirty_image_custom_output_path(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)
    dirty_imager = OskarDirtyImager(
        OskarDirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    with FileHandler() as tmp_dir:
        output_fits_path = os.path.join(
            tmp_dir,
            "test_dirty_image_custom_output_path.fits",
        )
        dirty_image = dirty_imager.create_dirty_image(vis, output_fits_path)

        assert dirty_image.data.ndim == 4
        assert os.path.exists(dirty_image.path)
        assert dirty_image.path == output_fits_path
