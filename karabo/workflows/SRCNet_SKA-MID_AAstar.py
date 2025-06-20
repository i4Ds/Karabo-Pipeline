"""
This script generates simulated visibilities and images resembling SKAO data.
It also outputs corresponding ObsCore metadata ready to be ingested to Rucio.

Images: dirty image and cleaned image using WSClean.
These are MFS images (frequency channels aggregated into one channel),
not full image cubes.
"""
# DON'T DO ANY API BREAKING CHANGES WITHOUT A GOOD REASON! There my be some folks
# relying on the stability of argparse and environment variable interface.

import math
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from typing import Tuple, cast

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
SKY_MODEL = Environment.get("SKY_MODEL", str)
PHASE_CENTER_RA_DEG = Environment.get("PHASE_CENTER_RA_DEG", float)
PHASE_CENTER_DEC_DEG = Environment.get("PHASE_CENTER_DEC_DEG", float)
START_FREQ_HZ = Environment.get("START_FREQ_HZ", float)
END_FREQ_HZ = Environment.get("END_FREQ_HZ", float)
FREQ_INC_HZ = Environment.get("FREQ_INC_HZ", float)
OBS_LENGTH_HOURS = Environment.get("OBS_LENGTH_HOURS", float)
# Original MIGHTEE_L1 survey: 3593 dumps => Size: 6668.534 GB
# SKA operations: 4 h blocks of cleaned data from SDP to SRCNet
NUM_TIME_STAMPS = Environment.get("NUM_TIME_STAMPS", int)
# During the chosen time range [start, start + length]
# sources shouldn't be behind horizon for obs-duration for
#   chosen sky, telescope & phase-center
# E.g. MIGHTEE-survey: 2020-04-26 14:36:50.820 UTC to 2020-04-26 22:35:42.665 UTC
START_DATE_AND_TIME = Environment.get(
    "START_DATE_AND_TIME", str
)  # UTC "2020-04-26T16:36"

# MeerKAT dish diameter = 13.5 m
# SKA-Mid dish diameter = 15 m
# AA*: 64*13.5 m + 80*15 m
ref_freq_hz = (START_FREQ_HZ + END_FREQ_HZ) / 2
wavelength_m = cast(float, constants.c.value / ref_freq_hz)
fov_rad = 1.25 * wavelength_m / 15
fov_deg = np.rad2deg(fov_rad)
filter_radius_deg = fov_deg * 3  # 3x primary beam

# Imaging
# Image size in degrees should be smaller than FOV
# Bigger baseline -> higher resolution
IMAGING_NPIXEL = Environment.get("IMAGING_NPIXEL", int)
# None calculates cellsize automatically
IMAGING_CELLSIZE = Environment.get(
    "IMAGING_CELLSIZE", float, None, allow_none_parsing=True
)
if IMAGING_CELLSIZE is None:
    IMAGING_CELLSIZE = fov_rad / IMAGING_NPIXEL
    print(f"Calculated {IMAGING_CELLSIZE=}")

# Rucio metadata
RUCIO_NAMESPACE = Environment.get("RUCIO_NAMESPACE", str)
RUCIO_LIFETIME = Environment.get("RUCIO_LIFETIME", int)

# ObsCore metadata
IVOID_AUTHORITY = Environment.get("IVOID_AUTHORITY", str)  # "test.skao"
IVOID_PATH = Environment.get("IVOID_PATH", str, "/~")
OBS_COLLECTION = Environment.get("OBS_COLLECTION", str, "SKAO/SKAMID")
OBS_ID = Environment.get("OBS_ID", str)  # provider unique observation-id

# files & dirs
FILE_PREFIX = Environment.get("FILE_PREFIX", str, "")  # for each file
OUT_DIR = Environment.get(
    "OUT_DIR", str
)  # ingestion-dir (don't create because it's persistent mounted volume)
if not os.path.exists(OUT_DIR):
    err_msg = f"f{OUT_DIR=} (ingestion-dir) doesn't exist!"
    raise FileNotFoundError(err_msg)
ingestion_dir = os.path.join(OUT_DIR, RUCIO_NAMESPACE)


def generate_visibilities(outdir: str) -> Visibility:
    simulator_backend = SimulatorBackend.OSKAR

    if SKY_MODEL == "MIGHTEE_L1":
        # https://archive.sarao.ac.za/search/MIGHTEE%20COSMOS/target/J0408-6545/captureblockid/1587911796/
        sky_model = SkyModel.get_MIGHTEE_Sky(
            min_freq=START_FREQ_HZ, max_freq=END_FREQ_HZ
        )
    else:
        err_msg = (
            f"Env-var {SKY_MODEL=} is not a valid value. Allowed values are: "
            + "`MIGHTEE_L1`."
        )
        raise ValueError(err_msg)
    print(
        f"{datetime.now()}: Infos in degree: RA={PHASE_CENTER_RA_DEG}, DEC={PHASE_CENTER_DEC_DEG}, FOV={fov_deg}"  # noqa: E501
    )
    print(
        f"{datetime.now()}: Filter sources outside of primary beam's sensitivity: {filter_radius_deg=}"  # noqa: E501
    )
    sky_model = sky_model.filter_by_radius(
        inner_radius_deg=0.0,
        outer_radius_deg=filter_radius_deg,
        ra0_deg=PHASE_CENTER_RA_DEG,
        dec0_deg=PHASE_CENTER_DEC_DEG,
    )

    telescope = Telescope.constructor(  # type: ignore[call-overload]
        name="SKA-MID-AAstar",
        version=SKAMidAAStarVersions.SKA_OST_ARRAY_CONFIG_2_3_1,
        backend=simulator_backend,
    )

    number_of_channels = math.floor((END_FREQ_HZ - START_FREQ_HZ) / FREQ_INC_HZ)
    use_gpus = is_cuda_available()
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=FREQ_INC_HZ,
        station_type="Gaussian beam",
        gauss_beam_fwhm_deg=fov_deg,
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

    os.makedirs(outdir, exist_ok=True)
    return simulation.run_simulation(  # type: ignore[no-any-return]
        telescope,
        sky_model,
        observation,
        backend=simulator_backend,
        visibility_path=os.path.join(
            outdir,
            f"{FILE_PREFIX}measurements.ms",
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


def create_dirty_image(visibility: Visibility, outdir: str) -> Image:
    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,  # type: ignore[arg-type]
            combine_across_frequencies=True,
        )
    )

    os.makedirs(outdir, exist_ok=True)
    return dirty_imager.create_dirty_image(
        visibility,
        output_fits_path=os.path.join(
            outdir,
            f"{FILE_PREFIX}dirty.fits",
        ),
    )


def create_cleaned_image(
    visibility: Visibility, dirty_image: Image, outdir: str
) -> Image:
    image_cleaner = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,  # type: ignore[arg-type]
        )
    )

    os.makedirs(outdir, exist_ok=True)
    return image_cleaner.create_cleaned_image(
        visibility,
        dirty_fits_path=dirty_image.path,
        output_fits_path=os.path.join(
            outdir,
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


def create_access_url(namespace: str, name: str) -> str:
    return f"https://datalink.ivoa.srcdev.skao.int/rucio/links?id={namespace}:{name}"


def parse_input() -> Tuple[bool, bool, bool]:
    parser = ArgumentParser(
        prog="SRCNet AAStar simulation",
        description="Produces visibility and images from simulation.",
    )
    parser.add_argument("--vis", "--visibility", "--visibilities", action="store_true")
    parser.add_argument("--clean", "--cleaned", action="store_true")
    parser.add_argument("--dirty", action="store_true")
    args = parser.parse_args()
    vis = args.vis
    clean = args.clean
    dirty = args.dirty
    return vis, dirty, clean


def main() -> None:
    vis, dirty, clean = parse_input()
    if not vis and not dirty and not clean:
        err_msg = "No data-product to create selected!"
        raise RuntimeError(err_msg)
    print(f"Producing: visibilities={vis}, dirty-img={dirty}, cleaned-img={clean}")

    with TemporaryDirectory() as tmpdir:
        print(f"{datetime.now()} Starting simulation")
        vis_out_dir = ingestion_dir if vis else tmpdir
        visibility = generate_visibilities(outdir=vis_out_dir)

        if vis:
            print(f"{datetime.now()} Creating visibility metadata")
            create_visibilities_metadata(visibility)

        if dirty or clean:
            print(f"{datetime.now()} Creating dirty image")
            dirty_out_dir = ingestion_dir if dirty else tmpdir
            dirty_image = create_dirty_image(visibility, outdir=dirty_out_dir)

            if dirty:
                print(f"{datetime.now()} Creating dirty image metadata")
                create_image_metadata(dirty_image)

            if clean:
                print(f"{datetime.now()} Creating cleaned image")
                cleaned_image = create_cleaned_image(
                    visibility, dirty_image, outdir=ingestion_dir
                )

                print(f"{datetime.now()} Creating cleaned image metadata")
                create_image_metadata(cleaned_image)

        print(f"{datetime.now()} Simulation workflow finished")


if __name__ == "__main__":
    main()
