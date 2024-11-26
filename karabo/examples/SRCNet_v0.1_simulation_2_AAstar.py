# This script generates simulated visibilities and images resembling SKAO data.
# It also outputs corresponding ObsCore metadata ready to be ingested to Rucio.
#
# Images: dirty image and cleaned image using WSClean.
# These are MFS images (frequency channels aggregated into one channel),
# not full image cubes.
#
# Size of generated data is around 3 TB:
# - 1.5 TB visibilities (before image cleaning)
# - 3 TB visibilities (after image cleaning)
# - 12 GB images
import math
import os
from datetime import datetime, timedelta, timezone

from karabo.data.obscore import ObsCoreMeta
from karabo.data.src import RucioMeta
from karabo.imaging.image import Image
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
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend
from karabo.util.helpers import get_rnd_str

# Simulation
# Phase center: should be mean of coverage
# Means of values from sky model description
PHASE_CENTER_RA = 150.12
PHASE_CENTER_DEC = 2.21

# Imaging
# Image size in degrees should be smaller than FOV
# Bigger baseline -> higher resolution
# Image resolution from SKAO's generate_visibilities.ipynb
IMAGING_NPIXEL = 20000
# -> Cellsize < FOV / 20000 -> 9.32190458333e-7
IMAGING_CELLSIZE = 9.3e-7

# Rucio
RUCIO_NAMESPACE = "chocolate"
# in seconds
RUCIO_LIFETIME = 31536000

# Metadata
OBS_COLLECTION = "SKAO/SKAMID"

obs_sim_id = 0  # inc/change for new simulation
user_rnd_str = get_rnd_str(k=7, seed=os.environ.get("USER"))
OBS_ID = f"karabo-{user_rnd_str}-{obs_sim_id}"  # unique ID per user & simulation

# Prefix for name part of Rucio DIDs.
# Example: visibilities DID will be [RUCIO_NAMESPACE]:[RUCIO_NAME_PREFIX]measurements.MS
# Keep in mind that DIDs need to be globally unique.
# Would probably make sense to make OBS_ID part of this for future runs.
RUCIO_NAME_PREFIX = "pi24_run_1_"

# Output root dir, this is just a default, set to your liking
OUTPUT_ROOT_DIR = os.path.join(os.environ["SCRATCH"], f"{RUCIO_NAME_PREFIX}output")
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=False)
print(f"Output will be written under output root dir {OUTPUT_ROOT_DIR}")


def generate_visibilities() -> Visibility:
    simulator_backend = SimulatorBackend.OSKAR

    # Link to metadata of survey:
    # https://archive.sarao.ac.za/search/MIGHTEE%20COSMOS/target/J0408-6545/captureblockid/1587911796/
    sky_model = SkyModel.get_MIGHTEE_Sky()

    telescope = Telescope.constructor(  # type: ignore[call-overload]
        name="SKA-MID-AAstar",
        version=SKAMidAAStarVersions.SKA_OST_ARRAY_CONFIG_2_3_1,
        backend=simulator_backend,
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
        phase_centre_ra_deg=PHASE_CENTER_RA,
        phase_centre_dec_deg=PHASE_CENTER_DEC,
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

    return simulation.run_simulation(  # type: ignore[no-any-return]
        telescope,
        sky_model,
        observation,
        backend=simulator_backend,
        visibility_path=os.path.join(
            OUTPUT_ROOT_DIR,
            f"{RUCIO_NAME_PREFIX}measurements.MS",
        ),
    )  # type: ignore[call-overload]


def create_visibilities_metadata(visibility: Visibility) -> None:
    ocm = ObsCoreMeta.from_visibility(
        vis=visibility,
        calibrated=False,
    )
    # remove path-infos for `name`
    name = os.path.split(visibility.path)[-1]
    rm = RucioMeta(
        namespace=RUCIO_NAMESPACE,  # needs to be specified by Rucio service
        name=name,
        lifetime=RUCIO_LIFETIME,
        dataset_name=None,
        meta=ocm,
    )
    # ObsCore mandatory fields
    ocm.obs_collection = OBS_COLLECTION
    ocm.obs_id = OBS_ID
    obs_publisher_did = RucioMeta.get_ivoid(  # rest args are defaults
        namespace=rm.namespace,
        name=rm.name,
    )
    ocm.obs_publisher_did = obs_publisher_did

    ocm.access_url = create_access_url(RUCIO_NAMESPACE, name)

    path_meta = RucioMeta.get_meta_fname(fname=visibility.path)
    _ = rm.to_dict(fpath=path_meta)
    print(f"Created {path_meta=}")


def create_dirty_image(visibility: Visibility) -> Image:
    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,
            combine_across_frequencies=True,
        )
    )

    return dirty_imager.create_dirty_image(
        visibility,
        output_fits_path=os.path.join(
            OUTPUT_ROOT_DIR,
            f"{RUCIO_NAME_PREFIX}dirty.fits",
        ),
    )


def create_cleaned_image(visibility: Visibility, dirty_image: Image) -> Image:
    image_cleaner = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,
        )
    )

    return image_cleaner.create_cleaned_image(
        visibility,
        dirty_fits_path=dirty_image.path,
        output_fits_path=os.path.join(
            OUTPUT_ROOT_DIR,
            f"{RUCIO_NAME_PREFIX}cleaned.fits",
        ),
    )


def create_image_metadata(image: Image) -> None:
    # Create image metadata
    ocm = ObsCoreMeta.from_image(img=image)
    # remove path-infos for `name`
    name = os.path.split(image.path)[-1]
    rm = RucioMeta(
        namespace=RUCIO_NAMESPACE,  # needs to be specified by Rucio service
        name=name,
        lifetime=RUCIO_LIFETIME,
        dataset_name=None,
        meta=ocm,
    )
    # ObsCore mandatory fields
    # some of the metadata is taken from the visibilities, since both data-products
    # originate from the same observation
    ocm.obs_collection = OBS_COLLECTION
    ocm.obs_id = OBS_ID
    obs_publisher_did = RucioMeta.get_ivoid(  # rest args are defaults
        namespace=rm.namespace,
        name=rm.name,
    )
    ocm.obs_publisher_did = obs_publisher_did

    ocm.s_ra = PHASE_CENTER_RA
    ocm.s_dec = PHASE_CENTER_DEC
    ocm.access_url = create_access_url(RUCIO_NAMESPACE, name)

    path_meta = RucioMeta.get_meta_fname(fname=image.path)
    _ = rm.to_dict(fpath=path_meta)
    print(f"Created {path_meta=}")


def create_access_url(namespace: str, name: str) -> str:
    return f"https://datalink.ivoa.srcdev.skao.int/rucio/links?id={namespace}:{name}"


if __name__ == "__main__":
    print(f"{datetime.now()} Script started")

    print(f"{datetime.now()} Starting simulation")
    visibility = generate_visibilities()

    print(f"{datetime.now()} Creating visibility metadata")
    create_visibilities_metadata(visibility)

    print(f"{datetime.now()} Creating dirty image")
    dirty_image = create_dirty_image(visibility)

    print(f"{datetime.now()} Creating dirty image metadata")
    create_image_metadata(dirty_image)

    print(f"{datetime.now()} Creating cleaned image")
    cleaned_image = create_cleaned_image(visibility, dirty_image)

    print(f"{datetime.now()} Creating cleaned image metadata")
    create_image_metadata(cleaned_image)

    print(f"{datetime.now()} Script finished")
