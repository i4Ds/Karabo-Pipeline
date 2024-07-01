"""ObsCore Data Model.

https://ivoa.net/documents/ObsCore/
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from warnings import warn

from karabo.util.file_handler import assert_valid_ending

_DataProductTypeType = Literal[
    "image",
    "cube",
    "spectrum",
    "sed",
    "timeseries",
    "visibility",
    "event",
    "measurements",
]
_CalibLevelType = Literal[0, 1, 2, 3, 4]


@dataclass
class ObsCoreMeta:
    """IVOA ObsCore metadata (TAP column names).

    This doesn't describe a full ObsCoreDM, but the mandatory and the ones
    defined in the SRCNet rucio database.

    Args:
        dataproduct_type: Logical data product type (image etc.). `image`, `cube`,
            `spectrum`, `sed`, `timeseries`, `visibility`, `event` or `measurements`.
        dataproduct_subtype: Data product specific type.
        calib_level: Calibration level {0, 1, 2, 3, 4} (NOT NULL).
            - 0: Raw instrumental data.
            - 1: Instrumental data in a starndard format (FITS, VOTable, etc.)
            - 2: Calibrated, science ready measurements without instrument signature.
            - 3: Enhanced data products like mosaics, drizzled images or heavily
                processed survey fields. May represent a combination of data from
                multiple primary obs.
            - 4: Analysis data products generated after scientific data manipulation.
        obs_collection: Name of the data collection (NOT NULL). Either registered
            shortname, full registered IVOA identifier or a data provider defined
            shortname. Often used pattern: `<facility-name>/<instrument-name>`.
        obs_id: Observation ID (NOT NULL).
            All data-products from a single observation should share the same `obs_id`.
            This is just a unique str-ID with no form. Must be unique to a provider.
        obs_publisher_did: Dataset identifier given by the publisher (NOT NULL).
            IVOA dataset identifier. Must be a unique value within the namespace
            controlled by the dataset publisher (data center).
        obs_title: Brief description of dataset in free format.
        obs_creator_did: IVOA dataset identifier given by the creator.
        target_class: Class of the Target object as in SSA.
        access_url: URL used to access (download) dataset.
        access_format: File content format (MIME type) (`fits`, `jpeg`, `zip`, etc.).
        access_estsize: [kbyte] Estimated size of dataset from `access_url`.
        target_name: Astronomical object observed, if any.
        s_ra: [deg] Central right ascension, ICRS.
        s_dec: [deg] Central declination, ICRS.
        s_fov: [deg] Diameter (bounds) of the covered region.
        s_region: Sky region covered by the data product (expressed in ICRS frame).
        s_resolution: [arcsec] Spatial resolution of data as FWHM.
        s_xel1: Number of elements along the first spatial axis.
        s_xel2: Number of elements along the second spatial axis.
        s_pixel_scale: Sampling period in world coordinate units along the spatial axis.
        t_min: [d] Start time in MJD.
        t_max: [d] Stop time in MJD.
        t_exptime: [s] Total exposure time.
        t_resolution: [s] Temporal resolution FWHM.
        t_xel: Number of elements along the time axis.
        em_min: [m] Start in spectral coordinates.
        em_max: [m] Stop in spectral coordinates.
        em_res_power: Spectral resolving power.
        em_xel: Number of elements along the spectral axis.
        em_ucd: Nature of the spectral axis.
        o_ucd: UCD of observable (e.g. phot.flux.density, phot.count, etc.)
        pol_states: List of polarization states or NULL if not applicable.
        pol_xel: Number of polarization samples.
        facility_name: Name of the facility used for this observation.
        instrument_name: Name of the instrument used for this observation.
        preview: TODO: couldn't find description in IVOA documentation.
    """

    dataproduct_type: _DataProductTypeType | None = None
    dataproduct_subtype: str | None = None
    calib_level: _CalibLevelType | None = None  # not null
    obs_collection: str | None = None  # not null
    obs_id: str | None = None  # not null
    obs_publisher_did: str | None = None  # not null
    obs_title: str | None = None
    obs_creator_did: str | None = None
    target_class: str | None = None
    access_url: str | None = None
    access_format: str | None = None
    access_estsize: int | None = None
    target_name: str | None = None
    s_ra: float | None = None
    s_dec: float | None = None
    s_fov: float | None = None
    s_region: str | None = None
    s_resolution: float | None = None
    s_xel1: int | None = None
    s_xel2: int | None = None
    s_pixel_scale: float | None = None
    t_min: float | None = None
    t_max: float | None = None
    t_exptime: float | None = None
    t_resolution: float | None = None
    t_xel: int | None = None
    em_min: float | None = None
    em_max: float | None = None
    em_res_power: float | None = None
    em_xel: int | None = None
    em_ucd: str | None = None
    o_ucd: str | None = None
    pol_states: str | None = None
    pol_xel: int | None = None
    facility_name: str | None = None
    instrument_name: str | None = None
    preview: str | None = None

    def to_json(
        self,
        fpath: Path | str,
        ignore_none: bool = True,
    ) -> None:
        """Converts this dataclass into a JSON.

        Args:
            fpath: JSON file-path.
            ignore_none: Ignore non-mandatory `None` fields?

        Returns:
            JSON as a str.
        """
        assert_valid_ending(path=fpath, ending=".json")
        dictionary = asdict(self)
        mandatory_fields = (
            "calib_level",
            "obs_collection",
            "obs_id",
            "obs_publisher_did",
        )
        mandatory_missing = [
            field_name
            for field_name in mandatory_fields
            if getattr(self, field_name) is None
        ]
        if len(mandatory_missing) > 0:
            wmsg = (
                f"{mandatory_missing=} fields are None in `ObsCoreMeta`, "
                + "but are mandatory to ObsTAP services."
            )
            warn(message=wmsg, category=UserWarning, stacklevel=1)
        if ignore_none:
            dictionary = {
                key: value
                for key, value in dictionary.items()
                if value is not None or key in mandatory_fields
            }
        with open(file=fpath, mode="w") as json_file:
            json_file.write(json.dumps(dictionary))
