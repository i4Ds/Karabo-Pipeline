import os
from datetime import datetime

from karabo.imaging.imager_rascil import (
    RascilDirtyImager,
    RascilDirtyImagerConfig,
    RascilImageCleaner,
    RascilImageCleanerConfig,
)
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles


def test_dirty_image(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty_imager = RascilDirtyImager(
        RascilDirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(vis)

    assert dirty_image.data.ndim == 4


def test_create_cleaned_image():
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
    visibility_askap = interferometer_sim.run_simulation(
        askap_tel,
        sky,
        observation_settings,
    )
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    # could fail if `xarray` and `ska-sdp-func-python` not compatible, see issue #542
    restored = RascilImageCleaner(
        RascilImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            ingest_vis_nchan=16,
            clean_nmajor=1,
            clean_algorithm="mmclean",
            clean_scales=[10, 30, 60],
            clean_threshold=0.12e-3,
            clean_nmoment=5,
            clean_psf_support=640,
            clean_restored_output="integrated",
            # TODO DASK_TEST_ISSUE Commented out to avoid test failure on GitHub
            # use_dask=True,
        )
    ).create_cleaned_image(visibility_askap)

    assert os.path.exists(restored.path)
