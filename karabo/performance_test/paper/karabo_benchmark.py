"""
This script is used for the performance tests in paper
Karabo: A versatile SKA Observation Simulation Framework.

Author of script: andreas.wassmer@fhnw.ch
"""
import argparse
import logging
import os
from datetime import datetime, timedelta

import numpy as np
import skymodel_reader

from karabo.imaging.backends.rascil_backend import (
    RascilBackendConfig,
    RascilBackendImager,
)
from karabo.imaging.imager_interface import ImageSpec
from karabo.imaging.imager_wsclean import WscleanImageCleaner, WscleanImageCleanerConfig
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.telescope import Telescope

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nc",
    default=1,
    help="number of frequency channels for simulation. Defaults to 1",
    type=int,
)
parser.add_argument(
    "-gpu", action="store_true", help="if given, use gpu for simulation"
)
args = parser.parse_args()

nchan = args.nc
use_gpus_ = False
if args.gpu:
    use_gpus_ = True

# --- Enable proper logging of parameters --- #
current_time = datetime.now()
current_time_str = current_time.strftime("%H%M%S")
basefilename = f"karabo_benchmark_cpu_nc_{nchan}_{current_time_str}"
if use_gpus_:
    basefilename = f"karabo_benchmark_gpu_nc_{nchan}_{current_time_str}"

logging.basicConfig(filename=f"{basefilename}.log")
logging.getLogger("func-python-logger").setLevel(logging.DEBUG)
logger = logging.getLogger("sim_simple_py")
logger.setLevel(logging.INFO)

# --- Code Cell for SKA-low and MWA Configuration --- #
telescope_path = "input/meerkat.tm"
gleam_sky_model_file_name = "input/GLEAM_EGC_v0.fits"
file_name = "gleam_meerkat_snapshot0"
path = os.path.join("output", basefilename)

noise_enable_ = False
enable_array_beam = False

img_nsize = 4096
cellsize_arcsec = 3  # beam_size_arcsec/3 # in arcsec
cellsize_rad = cellsize_arcsec / 3600.0 * np.pi / 180.0


logger.info(f"#### TELESCOPE PATH: {telescope_path}")
logger.info(f"#### File Name: {file_name}")

sky = skymodel_reader.read_gleam_sky_from_fits(gleam_sky_model_file_name)
sky.explore_sky(
    [250, -80],
    s=0.1,
    filename="output/gleam_paper.png",
    cbar_label="log$_{10}$ (Flux density (Jy))",
    cmap="jet",
    xlabel="Right Ascension (hours)",
    ylabel="Declination ($^o$)",
)

# ra_list = [292.5];dec_list = [-55.0]
ra_list = [150]
dec_list = [2.0]
f_obs = 8.56e8
chan = 0.2e6  # at SKA-mid frequencies
sec_ = 8
t_int = 8  # for SKA mid freq
hours_ = 0
min_ = 0  # 1 hour pointing
t_nchan = int((hours_ * 3600.0 + min_ * 60.0 + sec_) / t_int)

# will create about 97.000 sources
filter_radius = 11.0

phase_ra = ra_list[0]
phase_dec = dec_list[0]
sky_filter = sky.filter_by_radius(
    ra0_deg=phase_ra,
    dec0_deg=phase_dec,
    inner_radius_deg=0,
    outer_radius_deg=filter_radius,
)

sky_filter.explore_sky(
    [phase_ra, phase_dec], filename=os.path.join("output", "filtered_sky.png"), s=1
)
logger.info(f"Number of sources in GLEAM sky survey: {sky.sources.shape[0]}")
logger.info(
    f"                      After filtering: "
    f" {sky_filter.sources.shape[0]} ({filter_radius=})"
)
logger.info(f"              Number of time channels: {t_nchan}")
logger.info(f"         Number of freqeuncy channels: {nchan}")
logger.info(f"                         GPUs enabled: {use_gpus_}")

# ---------------------------------
layout = np.loadtxt(telescope_path + "/layout.txt")
nant = len(layout)
nb = int(nant * (nant - 1) * 0.5)
logger.info(f"Number of Baselines: {nb}")
bl = [0] * nb
k = 0
for i in range(nant):
    for j in range(i, nant):
        if i != j:
            bl[k] = np.sqrt(
                (layout[i][0] - layout[j][0]) ** 2 + (layout[i][1] - layout[j][1]) ** 2
            )
            k = k + 1

base_length = np.array(bl)
max_baseline = base_length.max()
beam_size_arcsec = 3.0e8 / f_obs / max_baseline * 180 / np.pi * 3600
logger.info(f"Maximum Baseline (m): {max_baseline}")
logger.info(f"Beam Size (arcsec): {beam_size_arcsec}")

start_time = datetime.now()
logger.info(f"#####----   START: {start_time.strftime('%H:%M:%S')}  ----##")
logger.info("##---------- RA-DEC Iteration begins -----------##")
k = 0
for phase_ra in ra_list:
    for phase_dec in dec_list:
        logger.info(f"Iteration: {k} | R.A.: {phase_ra} deg | DEC: {phase_dec} deg")
        # Skip analysis if there are no sources in this iteration
        if sky_filter.sources is None:
            continue
        telescope = Telescope.read_OSKAR_tm_file(telescope_path)
        telescope.read_OSKAR_tm_file(telescope_path)
        simulation = InterferometerSimulation(
            channel_bandwidth_hz=chan,
            time_average_sec=1,
            noise_enable=False,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=f_obs,
            noise_inc_freq=chan,
            noise_number_freq=1,
            noise_rms_start=5,
            noise_rms_end=10,
            use_gpus=use_gpus_,
        )
        observation = Observation(
            phase_centre_ra_deg=phase_ra,
            start_date_and_time=datetime(2022, 9, 1, 15, 00, 00, 521489),
            length=timedelta(hours=hours_, minutes=min_, seconds=sec_, milliseconds=0),
            phase_centre_dec_deg=phase_dec,
            number_of_time_steps=t_nchan,
            start_frequency_hz=f_obs,
            frequency_increment_hz=chan,
            number_of_channels=nchan,
        )

        logger.info("--- Simulation Run Begins....")
        visibility = simulation.run_simulation(
            telescope,
            sky_filter,
            observation,
            # visibility_path=path+'/'+file_name+str(k)+'.ms'
        )
        logger.info("--- Simulation Run Ends....")
        k = k + 1
        dirty_imager = RascilBackendImager(
            RascilBackendConfig(combine_across_frequencies=False)
        )
        image_spec = ImageSpec(
            npix=img_nsize,
            cellsize_arcsec=cellsize_arcsec,
            phase_centre_deg=(phase_ra, phase_dec),
        )

        dirty, _psf = dirty_imager.invert(visibility, image_spec)

        dirty.write_to_file(
            os.path.join(path, f"{file_name+str(k)}.fits"),
            overwrite=True,
        )


nchan_img = nchan
cellsize_arcsec = 15  # beam_size_arcsec/3 # in arcsec
cellsize_rad = cellsize_arcsec / 3600.0 * np.pi / 180.0
logger.info(
    f"BEAM SIZE: {beam_size_arcsec} arcsec | "
    f"IMAGE CELL SIZE: {cellsize_arcsec} arcsec | "
    f"IMG SIZE: {img_nsize} pixels | "
    f"IMG FOV: {img_nsize*cellsize_arcsec/3600} deg"
)

k = 0
logger.info(f"----OUTPUT MS PATH: {visibility.path} -----------")
output_img = os.path.join(path, "img", f"{file_name+str(k)}.ms")

logger.info(f"----OUTPUT IMG PATH: {output_img} -----------")
logger.info("##### IMAGING Begins.....")
wscleaner = WscleanImageCleaner(
    WscleanImageCleanerConfig(
        niter=5000,
        mgain=0.8,
        imaging_npixel=img_nsize,
        imaging_cellsize=cellsize_rad,
    )
)

cleaned_image = wscleaner.create_cleaned_image(visibility)
cleaned_image.write_to_file(os.path.join(path, "cleaned_image.fits"), overwrite=True)
cleaned_image.plot(filename=os.path.join(path, "cleaned_image.png"))

end_time = datetime.now()
logger.info(f"#####----  END: {end_time.strftime('%H:%M:%S')}  ----##")
logger.info(
    f"#####----  "
    f"DURATION: {(end_time - start_time).total_seconds() / 60} mins.  "
    "----#####"
)
