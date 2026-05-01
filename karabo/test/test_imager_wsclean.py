import math
import os
import shutil
import warnings

import pytest

from karabo.imaging.backends.wsclean_backend import WscleanBackendImager
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_factory import ImagingBackend, get_imager
from karabo.imaging.imager_interface import ImageSpec
from karabo.imaging.imager_wsclean import (
    WscleanDirtyImager,
    WscleanImageCleaner,
    WscleanImageCleanerConfig,
    create_image_custom_command,
)
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles
from karabo.util.file_handler import FileHandler
from karabo.warning import DIRECT_WSCLEAN_USAGE_MESSAGE, DirectWscleanUsageWarning

# this parameter sets the number of interations that WSClean does
# to clean an image. The default is set to 50_000. We don't need
# that many because we do run tests only, i.e. the quality of the
# result is not tested.
CLEAN_ITERATIONS = 100
WSCLEAN_AVAILABLE = shutil.which("wsclean") is not None


def test_wsclean_imager_factory_returns_adapter():
    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        imager = get_imager(ImagingBackend.WSCLEAN)
    assert not any(
        isinstance(warning.message, DirectWscleanUsageWarning)
        for warning in warning_record
    )
    assert isinstance(imager, WscleanBackendImager)


def test_direct_wsclean_dirty_imager_warns():
    with pytest.warns(DirectWscleanUsageWarning) as warning_record:
        WscleanDirtyImager(
            DirtyImagerConfig(
                imaging_npixel=64,
                imaging_cellsize=3.878509448876288e-05,
            ),
        )
    assert str(warning_record[0].message) == DIRECT_WSCLEAN_USAGE_MESSAGE


def test_direct_wsclean_cleaner_warns():
    with pytest.warns(DirectWscleanUsageWarning) as warning_record:
        WscleanImageCleaner(
            WscleanImageCleanerConfig(
                imaging_npixel=64,
                imaging_cellsize=3.878509448876288e-05,
                niter=CLEAN_ITERATIONS,
            ),
        )
    assert str(warning_record[0].message) == DIRECT_WSCLEAN_USAGE_MESSAGE


def test_direct_wsclean_custom_command_warns():
    with pytest.warns(DirectWscleanUsageWarning) as warning_record:
        with pytest.raises(ValueError):
            create_image_custom_command("not-wsclean")
    assert str(warning_record[0].message) == DIRECT_WSCLEAN_USAGE_MESSAGE


def test_wsclean_restore_before_invert_raises():
    imager = get_imager(ImagingBackend.WSCLEAN)
    with pytest.raises(RuntimeError, match="previous invert"):
        imager.restore(None, None)  # type: ignore[arg-type]


@pytest.mark.skipif(not WSCLEAN_AVAILABLE, reason="WSClean is not installed")
def test_wsclean_imager_factory_invert_and_restore_smoke(
    minimal_casa_ms: Visibility,
):
    imager = get_imager(ImagingBackend.WSCLEAN)
    spec = ImageSpec(
        npix=64,
        cellsize_arcsec=math.degrees(5e-5) * 3600.0,
        phase_centre_deg=(0.0, 0.0),
    )

    dirty_image, psf_image = imager.invert(minimal_casa_ms, spec)

    assert os.path.exists(dirty_image.path)
    assert os.path.exists(psf_image.path)
    assert dirty_image.data.size > 0
    assert psf_image.data.size > 0

    restored = imager.restore(dirty_image, psf_image)
    assert os.path.exists(restored.path)
    assert restored.data.size > 0


def test_dirty_image(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(vis)

    assert os.path.exists(dirty_image.path)


def test_dirty_image_custom_path(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)
    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    with FileHandler() as tmp_dir:
        output_fits_path = os.path.join(
            tmp_dir,
            "test_dirty_image_custom_path.fits",
        )
        dirty_image = dirty_imager.create_dirty_image(
            vis,
            output_fits_path=output_fits_path,
        )

        assert dirty_image.path == output_fits_path
        assert os.path.exists(dirty_image.path)


def test_create_cleaned_image(default_sample_simulation_visibility: Visibility):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            niter=CLEAN_ITERATIONS,
        )
    ).create_cleaned_image(default_sample_simulation_visibility)

    assert os.path.exists(restored.path)


def test_create_cleaned_image_custom_path(
    default_sample_simulation_visibility: Visibility,
):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    with FileHandler() as tmp_dir:
        output_fits_path = os.path.join(
            tmp_dir,
            "test_create_cleaned_image_custom_path.fits",
        )
        restored = WscleanImageCleaner(
            WscleanImageCleanerConfig(
                imaging_npixel=imaging_npixel,
                imaging_cellsize=imaging_cellsize,
                niter=CLEAN_ITERATIONS,
            )
        ).create_cleaned_image(
            default_sample_simulation_visibility,
            output_fits_path=output_fits_path,
        )

        assert restored.path == output_fits_path
        assert os.path.exists(restored.path)


def test_create_cleaned_image_reuse_dirty(
    default_sample_simulation_visibility: Visibility,
):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(default_sample_simulation_visibility)
    assert os.path.exists(dirty_image.path)

    restored = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            niter=CLEAN_ITERATIONS,
        )
    ).create_cleaned_image(
        default_sample_simulation_visibility,
        dirty_fits_path=dirty_image.path,
    )
    assert os.path.exists(dirty_image.path)
    assert os.path.exists(restored.path)


def test_create_image_custom_command(default_sample_simulation_visibility: Visibility):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored = create_image_custom_command(
        "wsclean "
        f"-size {imaging_npixel} {imaging_npixel} "
        f"-scale {math.degrees(imaging_cellsize)}deg "
        f"-niter {CLEAN_ITERATIONS} "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{default_sample_simulation_visibility.path}",
    )

    assert os.path.exists(restored.path)


def test_create_image_custom_command_multiple_outputs(
    default_sample_simulation_visibility: Visibility,
):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored, residual = create_image_custom_command(
        "wsclean "
        f"-size {imaging_npixel} {imaging_npixel} "
        f"-scale {math.degrees(imaging_cellsize)}deg "
        f"-niter {CLEAN_ITERATIONS} "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{default_sample_simulation_visibility.path}",
        ["wsclean-image.fits", "wsclean-residual.fits"],
    )  # type: ignore  # cannot infer type, should be List[Image]

    assert os.path.exists(restored.path)
    assert os.path.exists(residual.path)
