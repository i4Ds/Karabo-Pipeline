"""Example script to attach Rucio ObsCore metadata data-products for ingestion.

Here, we create a simulated visibilities and a cleaned image. This is just an example
script which can be highly adapted, e.g. you custom-sky, simulation params (for larger
data-products, simulation params as user-input, multiple simulations, etc.

Be aware that API-changes can take place in future Karabo-versions. This script is
based on `SRCNet_rucio_meta.py` but contains a full workflow for a single simulation.

If not specified further (e.g. through `XDG_CACHE_HOME` or `FileHandler.root_stm`),
Karabo is using /tmp. Thus if you have a script which is producing large and/or many
data products, we suggest to adapt the cache-root to a volume with more space.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from astropy import constants as const
from astropy import units as u

from karabo.data.obscore import FitsHeaderAxes, FitsHeaderAxis, ObsCoreMeta
from karabo.data.src import RucioMeta
from karabo.imaging.imager_wsclean import WscleanImageCleaner, WscleanImageCleanerConfig
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulator_backend import SimulatorBackend
from karabo.util.helpers import get_rnd_str


def main() -> None:
    # sky-to-visibilities simulation
    # the params for this example are highly adapted to not create large
    # data products, because this is not the focus of this script.
    sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=231e6)  # in Hz
    phase_center = [250, -80]  # RA,DEC in deg
    filter_radius_deg = 0.8
    sky = sky.filter_by_radius(0, filter_radius_deg, phase_center[0], phase_center[1])
    tel = Telescope.constructor("ASKAP", backend=SimulatorBackend.OSKAR)
    start_freq_hz = 76e6
    num_chan = 16
    freq_inc_hz = 1e8

    if num_chan < 1:
        err_msg = f"{num_chan=} but must be < 1"
        raise ValueError(err_msg)
    if num_chan == 1 and freq_inc_hz != 0:
        err_msg = f"{freq_inc_hz=} but must be 0 if only one channel is specified"
        raise ValueError(err_msg)

    obs = Observation(
        start_frequency_hz=start_freq_hz,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=num_chan,
        frequency_increment_hz=freq_inc_hz,
        number_of_time_steps=24,
    )
    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=freq_inc_hz)
    vis = interferometer_sim.run_simulation(
        telescope=tel,
        sky=sky,
        observation=obs,
        backend=SimulatorBackend.OSKAR,
    )
    # here, I customize the `vis.vis_path` to not have the same file-name for each
    # simulation which would be suboptimal for ingestion. Should probably be more
    # specific for your use-case.
    vis_path = os.path.join(
        os.path.split(vis.vis_path)[0],
        f"gleam-ra{phase_center[0]}-dec{phase_center[1]}.vis",
    )
    os.rename(
        vis.vis_path,
        vis_path,
    )
    vis.vis_path = vis_path

    # create metadata of visibility (currently [08-24], .vis supported, casa .ms not)
    vis_ocm = ObsCoreMeta.from_visibility(
        vis=vis,
        calibrated=False,
        tel=tel,
        obs=obs,
    )
    vis_rm = RucioMeta(
        namespace="testing",  # needs to be specified by Rucio service
        name=os.path.split(vis.vis_path)[-1],  # remove path-infos for `name`
        lifetime=86400,  # 1 day
        dataset_name=None,
        meta=vis_ocm,
    )
    # ObsCore mandatory fields
    vis_ocm.obs_collection = "MRO/ASKAP"
    obs_sim_id = 0  # nc for new simulation
    user_rnd_str = get_rnd_str(k=7, seed=os.environ.get("USER"))
    obs_id = f"karabo-{user_rnd_str}-{obs_sim_id}"  # unique ID per user & simulation
    vis_ocm.obs_id = obs_id
    obs_publisher_did = RucioMeta.get_ivoid(  # rest args are defaults
        namespace=vis_rm.namespace,
        name=vis_rm.name,
    )
    vis_ocm.obs_publisher_did = obs_publisher_did

    # fill/correct other fields of `ObsCoreMeta` here!
    # #####START#####
    # HERE
    # #####END#######

    vis_path_meta = RucioMeta.get_meta_fname(fname=vis.vis_path)
    _ = vis_rm.to_dict(fpath=vis_path_meta)
    print(f"Created {vis_path_meta=}")

    # -----Imaging-----

    mean_freq = start_freq_hz + freq_inc_hz * (num_chan - 1) / 2
    wavelength = const.c.value / mean_freq  # in m
    synthesized_beam = wavelength / tel.max_baseline()  # in rad
    imaging_cellsize = synthesized_beam / 3  # consider nyquist sampling theorem
    fov_deg = 2 * filter_radius_deg  # angular fov
    imaging_npixel_estimate = fov_deg / np.rad2deg(imaging_cellsize)  # not even&rounded
    imaging_npixel = int(np.floor((imaging_npixel_estimate + 1) / 2.0) * 2.0)

    print(f"Imaging: {imaging_npixel=}, {imaging_cellsize=} ...")
    restored = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            niter=5000,  # 10 times less than default
        )
    ).create_cleaned_image(  # currently, wsclean needs casa .ms, which is also created
        ms_file_path=vis.ms_file_path,
    )

    # customize fname for restored image for the same reason as visibilities
    restored_image_path = os.path.join(
        os.path.split(restored.path)[0],
        f"gleam-ra{phase_center[0]}-dec{phase_center[1]}.fits",
    )
    os.rename(
        restored.path,
        restored_image_path,
    )
    restored.path = restored_image_path

    # create metadata for restored .fits image
    # `FitsHeaderAxes` may need adaption based on the structure of your .fits image
    axes = FitsHeaderAxes(freq=FitsHeaderAxis(axis=3, unit=u.Hz))
    restored_ocm = ObsCoreMeta.from_image(img=restored, fits_axes=axes)
    restored_rm = RucioMeta(
        namespace="testing",  # needs to be specified by Rucio service
        name=os.path.split(restored.path)[-1],  # remove path-infos for `name`
        lifetime=86400,  # 1 day
        dataset_name=None,
        meta=restored_ocm,
    )
    # ObsCore mandatory fields
    # some of the metadata is taken from above, since both data-products originate
    # from the same observation
    restored_ocm.obs_collection = vis_ocm.obs_collection
    restored_ocm.obs_id = vis_ocm.obs_id
    obs_publisher_did = RucioMeta.get_ivoid(  # rest args are defaults
        namespace=restored_rm.namespace,
        name=restored_rm.name,
    )
    vis_ocm.obs_publisher_did = obs_publisher_did

    # fill/correct other fields of `ObsCoreMeta` here!
    # #####START#####
    # HERE
    # #####END#######

    restored_path_meta = RucioMeta.get_meta_fname(fname=restored.path)
    _ = restored_rm.to_dict(fpath=restored_path_meta)
    print(f"Created {restored_path_meta=}")


if __name__ == "__main__":
    main()
