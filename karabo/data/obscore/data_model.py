"""ObsCore Data Model.

https://ivoa.net/documents/ObsCore/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ObsCoreMeta:
    """IVOA ObsCore metadata (TAP column names).

    This doesn't describe a full ObsCoreDM, but the mandatory and the ones
    defined in the SRCNet rucio database.

    Args:
        dataproduct_type: Logical data product type (image etc.).
        dataproduct_subtype: Data product specific type.
        calib_level: Calibration level {0, 1, 2, 3, 4}.
        obs_collection: Name of the data collection.
        obs_id: Observation ID.
        obs_publisher_did: Dataset identifier given by the publisher.
        obs_title: Brief description of dataset in free format.
        obs_creator_did: IVOA dataset identifier given by the creator.
        target_class: Class of the Target object as in SSA.
        access_url: URL used to access (download) dataset.
        access_format: File content format (see BB.5.2).
        access_estsize: [kbyte] Estimated size of dataset in kilo bytes.
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

    dataproduct_type: str | None
    dataproduct_subtype: str | None
    calib_level: Literal[0, 1, 2, 3, 4]  # not null
    obs_collection: str  # not null
    obs_id: str  # not null
    obs_publisher_did: str  # not null
    obs_title: str | None
    obs_creator_did: str | None
    target_class: str | None
    access_url: str | None
    access_format: str | None
    access_estsize: int | None
    target_name: str | None
    s_ra: float | None
    s_dec: float | None
    s_fov: float | None
    s_region: str | None
    s_resolution: float | None
    s_xel1: int | None
    s_xel2: int | None
    s_pixel_scale: float | None
    t_min: float | None
    t_max: float | None
    t_exptime: float | None
    t_resolution: float | None
    t_xel: int | None
    em_min: float | None
    em_max: float | None
    em_res_power: float | None
    em_xel: int | None
    em_ucd: str | None
    o_ucd: str | None
    pol_states: str | None
    pol_xel: int | None
    facility_name: str | None
    instrument_name: str | None
    preview: str | None
