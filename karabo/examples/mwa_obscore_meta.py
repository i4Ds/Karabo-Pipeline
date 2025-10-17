#!/usr/bin/env python3
"""Script to generate ObsCore metadata for MWA UVFITS files.

This script demonstrates how to:
1. Load MWA UVFITS files using pyuvdata
2. Generate ObsCore metadata
3. Create the .meta JSON file required for ingestion
"""

import os
import math
from typing import Optional
from pprint import pprint

import numpy as np
from astropy import constants as const
from pyuvdata import UVData

from karabo.data.obscore import ObsCoreMeta
from karabo.data.src import RucioMeta
from karabo.simulation.visibility import Visibility
from karabo.util.helpers import get_rnd_str


def generate_mwa_metadata(
    uvfits_path: str,
    *,
    namespace: str = "MWA",
    lifetime: int = 31536000,  # 1 year in seconds
    obs_collection: str = "MWA",
    obs_id: Optional[str] = None,
) -> None:
    """Generate ObsCore metadata for an MWA UVFITS file.

    Args:
        uvfits_path: Path to UVFITS file
        namespace: Rucio namespace
        lifetime: Lifetime in seconds for the data in Rucio
        obs_collection: Observatory/instrument collection name
        obs_id: Optional observation ID. If None, will generate one.
    """
    # Load the UVFITS file using pyuvdata
    uv = UVData()
    uv.read_uvfits(uvfits_path)

    # Create visibility object
    vis = Visibility(uvfits_path)

    # Create ObsCore metadata
    ocm = ObsCoreMeta.from_visibility(
        vis=vis,
        calibrated=False,  # Set to True if calibrated data
    )

    # Set mandatory fields
    ocm.obs_collection = obs_collection

    if obs_id is None:
        # Generate unique observation ID if not provided
        user_rnd_str = get_rnd_str(k=7, seed=os.environ.get("USER"))
        obs_id = f"mwa-{user_rnd_str}"
    ocm.obs_id = obs_id

    # Create Rucio metadata object
    rm = RucioMeta(
        namespace=namespace,
        name=os.path.split(vis.path)[-1],  # Remove path info
        lifetime=lifetime,
        dataset_name=None,
        meta=ocm,
    )

    # Set publisher DID
    obs_publisher_did = RucioMeta.get_ivoid(
        namespace=rm.namespace,
        name=rm.name,
    )
    ocm.obs_publisher_did = obs_publisher_did

    # Write metadata to JSON file
    meta_path = RucioMeta.get_meta_fname(fname=uvfits_path)
    d = rm.to_dict(fpath=meta_path)
    print(f"Created metadata file: {meta_path}")
    pprint(d)

    breakpoint()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ObsCore metadata for MWA UVFITS files"
    )
    parser.add_argument("uvfits_path", help="Path to UVFITS file")
    parser.add_argument("--namespace", default="MWA", help="Rucio namespace")
    parser.add_argument(
        "--lifetime",
        type=int,
        default=31536000,
        help="Lifetime in seconds for the data in Rucio",
    )
    parser.add_argument(
        "--obs-collection", default="MWA", help="Observatory/instrument collection name"
    )
    parser.add_argument("--obs-id", help="Optional observation ID")

    args = parser.parse_args()

    generate_mwa_metadata(
        uvfits_path=args.uvfits_path,
        namespace=args.namespace,
        lifetime=args.lifetime,
        obs_collection=args.obs_collection,
        obs_id=args.obs_id,
    )
