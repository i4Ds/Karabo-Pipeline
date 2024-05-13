import os
from datetime import datetime

from karabo.imaging.imager_base import DirtyImagerConfig, ImageCleanerConfig
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


def test_constructor_from_dirty_imager_config():
    imaging_npixel = 1024
    imaging_cellsize = 0.1
    combine_across_frequencies = False

    dirty_imager = RascilDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            combine_across_frequencies=combine_across_frequencies,
        )
    )
    assert isinstance(dirty_imager.config, RascilDirtyImagerConfig)
    assert dirty_imager.config.imaging_npixel == imaging_npixel
    assert dirty_imager.config.imaging_cellsize == imaging_cellsize
    assert dirty_imager.config.combine_across_frequencies == combine_across_frequencies

    dirty_imager = RascilDirtyImager(
        RascilDirtyImagerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            combine_across_frequencies=combine_across_frequencies,
            override_cellsize=False,
        )
    )
    assert isinstance(dirty_imager.config, RascilDirtyImagerConfig)
    assert dirty_imager.config.imaging_npixel == imaging_npixel
    assert dirty_imager.config.imaging_cellsize == imaging_cellsize
    assert dirty_imager.config.combine_across_frequencies == combine_across_frequencies


def test_dirty_image(tobject: TFiles):
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)

    dirty_imager = RascilDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(vis)

    assert dirty_image.data.ndim == 4


def test_constructor_from_image_cleaner_config():
    imaging_npixel = 1024
    imaging_cellsize = 0.1

    image_cleaner = RascilImageCleaner(
        ImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
        )
    )
    assert isinstance(image_cleaner.config, RascilImageCleanerConfig)
    assert image_cleaner.config.imaging_npixel == imaging_npixel
    assert image_cleaner.config.imaging_cellsize == imaging_cellsize

    image_cleaner = RascilImageCleaner(
        RascilImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
        )
    )
    assert isinstance(image_cleaner.config, RascilImageCleanerConfig)
    assert image_cleaner.config.imaging_npixel == imaging_npixel
    assert image_cleaner.config.imaging_cellsize == imaging_cellsize


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
            use_dask=True,
        )
    ).create_cleaned_image(
        ms_file_path=visibility_askap.ms_file_path,
    )

    assert os.path.exists(restored.path)
