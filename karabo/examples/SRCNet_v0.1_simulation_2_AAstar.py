# This script generates simulated visibilities and dirty images resembling SKAO data.
# Size of generated data is around 1.6 TB.
import math
from datetime import datetime, timedelta, timezone

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_wsclean import (
    WscleanDirtyImager,
    WscleanImageCleaner,
    WscleanImageCleanerConfig,
)
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.telescope_versions import SKAMidAAStarVersions
from karabo.simulator_backend import SimulatorBackend

if __name__ == "__main__":
    print(f"{datetime.now()} Script started")

    # Simulate using OSKAR
    SIMULATOR_BACKEND = SimulatorBackend.OSKAR

    # Phase center: should be mean of coverage
    # Link to metadata of survey:
    # https://archive.sarao.ac.za/search/MIGHTEE%20COSMOS/target/J0408-6545/captureblockid/1587911796/
    sky_model = SkyModel.get_MIGHTEE_Sky()
    # Means of values from sky model description
    phase_center_ra = 150.12
    phase_center_dec = 2.21

    telescope = Telescope.constructor(  # type: ignore[call-overload]
        name="SKA-MID-AAstar",
        version=SKAMidAAStarVersions.SKA_OST_ARRAY_CONFIG_2_3_1,
        backend=SIMULATOR_BACKEND,
    )

    # From sky model description
    start_frequency_hz = 1.304e9
    end_frequency_hz = 1.375e9

    # From survey metadata
    frequency_increment_hz = 26123

    # Original survey: 32768 channels over the full frequency range
    # of 856 MHz to 1712 MHz
    number_of_channels = math.floor(
        (end_frequency_hz - start_frequency_hz) / frequency_increment_hz
    )
    print(f"{datetime.now()} number_of_channels={number_of_channels}")

    # Original survey: 3593 dumps => Size: 6668.534 GB
    # Observation time: 8 h
    # SKA operations: 4 h blocks of cleaned data from SDP to SRCNet
    number_of_time_steps = 1800

    # Wavelength 1340 MHz = 0.22372571 m
    # MeerKAT dish diameter = 13.5 m
    # SKA-Mid dish diameter = 15 m
    # AA*: 64*13.5 m + 80*15 m
    # 1.25 factor according to SKAO's yitl_observatory_data_rates.ipynb
    # FOV = Beam Width (FWHM) = 1.25 * 0.22372571 m / 15 m
    # = 0.01864380916666666666666666666667 rad = 1.0682115792999662 deg
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=frequency_increment_hz,
        station_type="Gaussian beam",
        gauss_beam_fwhm_deg=1.0682115792999662,
        gauss_ref_freq_hz=1.34e9,
        use_gpus=True,
    )

    observation = Observation(
        phase_centre_ra_deg=phase_center_ra,
        phase_centre_dec_deg=phase_center_dec,
        # During the chosen time range [start, start + length]
        # sources shouldn't be behind horizon, otherwise we won't see much.
        # Original survey: 2020-04-26 14:36:50.820 UTC to 2020-04-26 22:35:42.665 UTC
        start_date_and_time=datetime(2020, 4, 26, 16, 36, 0, 0, timezone.utc),
        # Dump rate from survey metadata
        length=timedelta(seconds=number_of_time_steps * 7.997),
        number_of_time_steps=number_of_time_steps,
        number_of_channels=number_of_channels,
        start_frequency_hz=start_frequency_hz,
        frequency_increment_hz=frequency_increment_hz,
    )

    print(f"{datetime.now()} Starting simulation")
    visibility = simulation.run_simulation(
        telescope,
        sky_model,
        observation,
        backend=SIMULATOR_BACKEND,
    )  # type: ignore[call-overload]

    # Imaging using WSClean
    # Image size in degrees should be smaller than FOV
    # Bigger baseline -> higher resolution
    # Image resolution from SKAO's generate_visibilities.ipynb
    imaging_npixel = 20000
    # -> Cellsize < FOV / 20000 -> 9.32190458333e-7
    imaging_cellsize = 9.3e-7

    print(f"{datetime.now()} Creating dirty image")
    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            combine_across_frequencies=True,
        )
    )
    dirty_image = dirty_imager.create_dirty_image(visibility)

    print(f"{datetime.now()} Creating cleaned image")
    image_cleaner = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
        )
    )
    cleaned_image = image_cleaner.create_cleaned_image(
        visibility,
        dirty_fits_path=dirty_image.path,
    )

    print(f"{datetime.now()} Script finished")
