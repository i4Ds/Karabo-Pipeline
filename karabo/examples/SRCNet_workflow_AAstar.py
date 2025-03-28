# This script generates simulated visibilities and images resembling SKAO data.
# It also outputs corresponding ObsCore metadata ready to be ingested to Rucio.
#
# Images: dirty image and cleaned image using WSClean.
# These are MFS images (frequency channels aggregated into one channel),
# not full image cubes.
#
# Size of generated data with default settings should be around 3 TB:
# - 1.5 TB visibilities (before image cleaning)
# - 3 TB visibilities (after image cleaning)
# - 12 GB images
import math
import os
from datetime import datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd
from astropy import constants

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
from karabo.util.gpu_util import is_cuda_available
from karabo.util.helpers import Environment

# Simulation
# Phase center: Must be inside coverage region of selected sky
SKY_MODEL = Environment.get("SKY_MODEL", str, "MIGHTEE_L1")
PHASE_CENTER_RA_DEG = Environment.get("PHASE_CENTER_RA_DEG", float, 150.12)
PHASE_CENTER_DEC_DEG = Environment.get("PHASE_CENTER_DEC_DEG", float, 2.21)
START_FREQ_HZ = Environment.get("START_FREQ_HZ", float, 1.304e9)
END_FREQ_HZ = Environment.get("END_FREQ_HZ", float, 1.375e9)
FREQ_INC_HZ = Environment.get("FREQ_INC_HZ", float, 26123.0)
OBS_LENGTH_HOURS = Environment.get("OBS_LENGTH_HOURS", float, 4.0)
# Original survey: 3593 dumps => Size: 6668.534 GB
# Observation time: 8 h
# SKA operations: 4 h blocks of cleaned data from SDP to SRCNet
NUM_TIME_STAMPS = Environment.get("NUM_TIME_STAMPS", int, 1800)
# During the chosen time range [start, start + length]
# sources shouldn't be behind horizon for obs-duration for
#   chosen sky, telescope & phase-center
# E.g. MIGHTEE-survey: 2020-04-26 14:36:50.820 UTC to 2020-04-26 22:35:42.665 UTC
START_DATE_AND_TIME = Environment.get(  # UTC
    "START_DATE_AND_TIME", str, "04.26.2020T16:36"
)

# Wavelength at e.g. 1340 MHz = 0.22372571 m
# MeerKAT dish diameter = 13.5 m
# SKA-Mid dish diameter = 15 m
# AA*: 64*13.5 m + 80*15 m
# 1.25 factor according to SKAO's yitl_observatory_data_rates.ipynb
# fov-deg = Beam Width (FWHM) = 1.25 * 0.22372571 m / 15 m * 180/pi
ref_freq_hz = (START_FREQ_HZ + END_FREQ_HZ) / 2
wavelength_m = cast(float, constants.c.value / ref_freq_hz)
fov_rad = 1.25 * wavelength_m / 15

# Imaging
# Image size in degrees should be smaller than FOV
# Bigger baseline -> higher resolution
# Image resolution from SKAO's generate_visibilities.ipynb
IMAGING_NPIXEL = Environment.get("IMAGING_NPIXEL", int, 20000)
# None calculates cellsize automatically
IMAGING_CELLSIZE = Environment.get("IMAGING_CELLSIZE", float, None)
if IMAGING_CELLSIZE is None:
    IMAGING_CELLSIZE = fov_rad / IMAGING_NPIXEL
    print(f"Calculated {IMAGING_CELLSIZE=}")

# Rucio metadata
RUCIO_NAMESPACE = Environment.get("RUCIO_NAMESPACE", str, "testing")
RUCIO_LIFETIME = Environment.get("RUCIO_LIFETIME", int, 31536000)  # [s]

# ObsCore metadata
IVOID_AUTHORITY = Environment.get("IVOID_AUTHORITY", str, "test.skao")
IVOID_PATH = Environment.get("IVOID_PATH", str, "/~")
OBS_COLLECTION = Environment.get("OBS_COLLECTION", str, "SKAO/SKAMID")
OBS_ID = Environment.get("OBS_ID", str)  # provider unique observation-id

# files & dirs
FILE_PREFIX = Environment.get("FILE_PREFIX", str, "")  # for each file
OUT_DIR = Environment.get("OUT_DIR", str, OBS_ID)
if OUT_DIR == OBS_ID and "/" in OBS_ID:
    err_msg = f"{OBS_ID=} is used as `OUT_DIR` (default), but `OBS_ID` contains `/`."
    raise ValueError(err_msg)
os.makedirs(OUT_DIR, exist_ok=False)
print(f"Write output in {os.path.abspath(OUT_DIR)} ...")


def generate_visibilities() -> Visibility:
    simulator_backend = SimulatorBackend.OSKAR

    if SKY_MODEL == "MIGHTEE_L1":
        # https://archive.sarao.ac.za/search/MIGHTEE%20COSMOS/target/J0408-6545/captureblockid/1587911796/
        sky_model = SkyModel.get_MIGHTEE_Sky(
            min_freq=START_FREQ_HZ, max_freq=END_FREQ_HZ
        )
    elif SKY_MODEL == "GLEAM":
        sky_model = SkyModel.get_GLEAM_Sky(min_freq=START_FREQ_HZ, max_freq=END_FREQ_HZ)
    elif SKY_MODEL == "MALS_DR1V3":
        sky_model = SkyModel.get_MALS_DR1V3_Sky(
            min_freq=START_FREQ_HZ, max_freq=END_FREQ_HZ
        )
    else:
        err_msg = (
            f"Env-var {SKY_MODEL=} is not a valid value. Allowed values are: "
            + "`MIGHTEE_L1`, `GLEAM` or `MALS_DR1V3`."
        )
        raise ValueError(err_msg)

    telescope = Telescope.constructor(  # type: ignore[call-overload]
        name="SKA-MID-AAstar",
        version=SKAMidAAStarVersions.SKA_OST_ARRAY_CONFIG_2_3_1,
        backend=simulator_backend,
    )

    # Original survey: 32768 channels over the full frequency range
    # of 856 MHz to 1712 MHz
    number_of_channels = math.floor((END_FREQ_HZ - START_FREQ_HZ) / FREQ_INC_HZ)
    use_gpus = is_cuda_available()
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=FREQ_INC_HZ,
        station_type="Gaussian beam",
        gauss_beam_fwhm_deg=np.rad2deg(fov_rad),
        gauss_ref_freq_hz=ref_freq_hz,
        use_gpus=use_gpus,
    )

    start_date_and_time = pd.to_datetime(START_DATE_AND_TIME, utc=True).to_pydatetime()
    observation = Observation(
        phase_centre_ra_deg=PHASE_CENTER_RA_DEG,
        phase_centre_dec_deg=PHASE_CENTER_DEC_DEG,
        start_date_and_time=start_date_and_time,
        length=timedelta(hours=OBS_LENGTH_HOURS),
        number_of_time_steps=NUM_TIME_STAMPS,
        number_of_channels=number_of_channels,
        start_frequency_hz=START_FREQ_HZ,
        frequency_increment_hz=FREQ_INC_HZ,
    )

    return simulation.run_simulation(  # type: ignore[no-any-return]
        telescope,
        sky_model,
        observation,
        backend=simulator_backend,
        visibility_path=os.path.join(
            OUT_DIR,
            f"{FILE_PREFIX}measurements.MS",
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
            imaging_cellsize=IMAGING_CELLSIZE,  # type: ignore[arg-type]
            combine_across_frequencies=True,
        )
    )

    return dirty_imager.create_dirty_image(
        visibility,
        output_fits_path=os.path.join(
            OUT_DIR,
            f"{FILE_PREFIX}dirty.fits",
        ),
    )


def create_cleaned_image(visibility: Visibility, dirty_image: Image) -> Image:
    image_cleaner = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,  # type: ignore[arg-type]
        )
    )

    return image_cleaner.create_cleaned_image(
        visibility,
        dirty_fits_path=dirty_image.path,
        output_fits_path=os.path.join(
            OUT_DIR,
            f"{FILE_PREFIX}cleaned.fits",
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

    ocm.s_ra = PHASE_CENTER_RA_DEG
    ocm.s_dec = PHASE_CENTER_DEC_DEG
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
