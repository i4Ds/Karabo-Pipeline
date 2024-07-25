"""Example script to attach Rucio ObsCore metadata data-products for ingestion.

This script probably needs adaption for your use-case. The parameters e.g. are not
    customizable through an API. It also assumes that there's already a visibility
    and image file available. Otherwise, you have to create them first. The manually
    added values in this script are arbitrary to some extent and should be set (or not)
    by yourself.

An end-to-end workflow would add the needed parts at the end of its simulation.
    However, we just operate on existing files to avoid example-duplication.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from datetime import datetime

from astropy import units as u

from karabo.data.obscore import FitsHeaderAxes, FitsHeaderAxis, ObsCoreMeta
from karabo.data.src import RucioMeta
from karabo.imaging.image import Image
from karabo.simulation.observation import Observation
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.util.helpers import get_rnd_str


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--dp-path",
        required=True,
        type=str,
        help="Path to data product inode (most likely file).",
    )
    parser.add_argument(
        "--dp-type",
        required=True,
        type=str,
        choices=["image", "visibility"],
        help="Data product type. See `ObsCoreMeta` which file-formats are supported.",
    )
    args = parser.parse_args()
    dp_path: str = args.dp_path
    dp_type: str = args.dp_type

    if not os.path.exists(dp_path):
        err_msg = f"Inode {dp_path=} doesn't exist!"
        raise RuntimeError(err_msg)
    dp_path_meta = RucioMeta.get_meta_fname(fname=dp_path)
    if os.path.exists(dp_path_meta):
        err_msg = f"{dp_path_meta=} already exists!"
        raise FileExistsError(err_msg)
    if dp_type == "image":
        image = Image(path=dp_path)
        # `FitsHeaderAxes` may need adaption based on the structure of your .fits image
        axes = FitsHeaderAxes(freq=FitsHeaderAxis(axis=4, unit=u.Hz))
        ocm = ObsCoreMeta.from_image(img=image, fits_axes=axes)
    elif dp_type == "visibility":
        vis = Visibility(vis_path=dp_path)  # .vis supported, .ms not atm [07/2024]
        # To extract additional information, `Telescope` & `Observation` should be
        # provided with the same settings as `vis` was created. As mentioned in the
        # module docstring, this is only necessary because we don't show the whole
        # workflow here.
        telescope = Telescope.constructor("ASKAP")
        observation = Observation(  # settings from notebook, of `minimal_visibility`
            start_frequency_hz=100e6,
            start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
            phase_centre_ra_deg=250.0,
            phase_centre_dec_deg=-80.0,
            number_of_channels=16,
            frequency_increment_hz=1e6,
            number_of_time_steps=24,
        )
        ocm = ObsCoreMeta.from_visibility(
            vis=vis,
            calibrated=False,
            tel=telescope,
            obs=observation,
        )
    else:
        err_msg = f"Unexpected {dp_type=}, allowed are only `dp-type` choices."
        raise RuntimeError(err_msg)

    # adapt each field according to your needs

    # be sure that name & namespace together are unique, e.g. by having different fnames
    name = os.path.split(dp_path)[-1]
    rm = RucioMeta(
        namespace="testing",  # needs to be specified by Rucio service
        name=name,
        lifetime=86400,  # 1 day
        dataset_name=None,
        meta=ocm,
    )

    # ObsCore mandatory fields
    ocm.obs_collection = "MRO/ASKAP"
    obs_sim_id = 0  # unique observation-simulation ID of `USER`
    user_rnd_str = get_rnd_str(k=7, seed=os.environ.get("USER"))
    ocm.obs_id = f"karabo-{user_rnd_str}-{obs_sim_id}"
    obs_publisher_did = RucioMeta.get_ivoid(  # rest args are defaults
        namespace=rm.namespace,
        name=rm.name,
    )
    ocm.obs_publisher_did = obs_publisher_did

    # fill other fields of `ObsCoreMeta` here!
    # #####START#####
    # HERE
    # #####END#######

    _ = rm.to_dict(fpath=dp_path_meta)
    print(f"Created {dp_path_meta}")


if __name__ == "__main__":
    main()
