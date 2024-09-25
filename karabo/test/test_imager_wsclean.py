import math
import os
from datetime import datetime

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_wsclean import (
    WscleanDirtyImager,
    WscleanImageCleaner,
    WscleanImageCleanerConfig,
    create_image_custom_command,
)
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles
from karabo.util.file_handler import FileHandler


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
        dirty_image = dirty_imager.create_dirty_image(vis, output_fits_path)

        assert dirty_image.path == output_fits_path
        assert os.path.exists(dirty_image.path)


def _run_sim() -> Visibility:
    phase_center = [250, -80]
    gleam_sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)
    sky = gleam_sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)
    askap_tel = Telescope.constructor("ASKAP")
    observation_settings = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=16,
        number_of_time_steps=24,
    )

    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)

    return interferometer_sim.run_simulation(
        askap_tel,
        sky,
        observation_settings,
    )


def test_create_cleaned_image():
    visibility = _run_sim()

    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
        )
    ).create_cleaned_image(visibility)

    assert os.path.exists(restored.path)


def test_create_cleaned_image_custom_path():
    visibility = _run_sim()

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
        ).create_cleaned_image(visibility, output_fits_path=output_fits_path)

        assert restored.path == output_fits_path
        assert os.path.exists(restored.path)


def test_create_cleaned_image_reuse_dirty():
    visibility = _run_sim()

    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(visibility)

    assert os.path.exists(dirty_image.path)

    restored = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
        )
    ).create_cleaned_image(visibility, dirty_fits_path=dirty_image.path)

    assert os.path.exists(dirty_image.path)
    assert os.path.exists(restored.path)


def test_create_image_custom_command():
    visibility = _run_sim()

    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored = create_image_custom_command(
        "wsclean "
        f"-size {imaging_npixel} {imaging_npixel} "
        f"-scale {math.degrees(imaging_cellsize)}deg "
        "-niter 50000 "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{visibility.path}"
    )

    assert os.path.exists(restored.path)


def test_create_image_custom_command_multiple_outputs():
    visibility = _run_sim()

    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored, residual = create_image_custom_command(
        "wsclean "
        f"-size {imaging_npixel} {imaging_npixel} "
        f"-scale {math.degrees(imaging_cellsize)}deg "
        "-niter 50000 "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{visibility.path}",
        ["wsclean-image.fits", "wsclean-residual.fits"],
    )

    assert os.path.exists(restored.path)
    assert os.path.exists(residual.path)
