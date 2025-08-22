import math
import os

import pytest

# from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_wsclean import (
    WscleanDirtyImager,
    WscleanDirtyImagerConfig,
    WscleanImageCleaner,
    WscleanImageCleanerConfig,
    create_image_custom_command,
)
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles
from karabo.util.file_handler import FileHandler


def test_dirty_image(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty_imager = WscleanDirtyImager(
        WscleanDirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(vis)

    assert os.path.exists(dirty_image.path)


def test_dirty_image_series(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    NUM_TIME_STEPS = 24
    dirty_imager = WscleanDirtyImager(
        WscleanDirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
            intervals_out=NUM_TIME_STEPS,  # time steps in gleam ms
        ),
    )
    dirty_image_list = dirty_imager.create_dirty_image_series(vis)
    assert dirty_image_list
    assert len(dirty_image_list) == 24


def test_intervals_set_should_raise_exception(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty_imager = WscleanDirtyImager(
        WscleanDirtyImagerConfig(
            imaging_npixel=2048, imaging_cellsize=3.878509448876288e-05, intervals_out=1
        ),
    )

    with pytest.raises(NotImplementedError) as excinfo:
        _ = dirty_imager.create_dirty_image(vis)
    assert "The parameter '-intervals-out'" in str(excinfo.value)


def test_intervals_not_set_should_raise_exception(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty_imager = WscleanDirtyImager(
        WscleanDirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        _ = dirty_imager.create_dirty_image_series(vis)
    assert "You must set the parameter" in str(excinfo.value)


def test_dirty_image_custom_path(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)
    dirty_imager = WscleanDirtyImager(
        WscleanDirtyImagerConfig(
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
        WscleanDirtyImagerConfig(
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
        "-niter 50000 "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{default_sample_simulation_visibility.path}"
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
        "-niter 50000 "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{default_sample_simulation_visibility.path}",
        ["wsclean-image.fits", "wsclean-residual.fits"],
    )

    assert os.path.exists(restored.path)
    assert os.path.exists(residual.path)
