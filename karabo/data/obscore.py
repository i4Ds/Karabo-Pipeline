"""ObsCore Data Model.

https://ivoa.net/documents/ObsCore/

Recommended IVOA documents: https://www.ivoa.net/documents/index.html
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    overload,
)
from warnings import warn

import numpy as np
from oskar import VisHeader
from typing_extensions import TypeGuard, assert_never

from karabo.imaging.image import Image
from karabo.simulation.observation import Observation
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.util._types import FilePathType
from karabo.util.file_handler import assert_valid_ending, getsize

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
_PolStatesType = Literal[
    "I",
    "Q",
    "U",
    "V",
    "RR",
    "LL",
    "RL",
    "LR",
    "XX",
    "YY",
    "XY",
    "YX",
    "POLI",
    "POLA",
]
_PolStatesListType = List[_PolStatesType]


@dataclass  # once when just Python >= 3.10 is supported, change to (kw_only=True)
class ObsCoreMeta:
    r"""IVOA ObsCore v1.1 metadata (TAP column names).

    This doesn't describe a full ObsCoreDM, but the mandatory and the non-mandatory
    fields defined in the SRCNet rucio database. The actual JSON to send to a specific
    ObsTAP service has to be created by yourself.

    The args-docstring provides just a rough idea of the according values. A
    more detailed description is provided by the ObsCore-v1.1 documentation.

    Args:
        dataproduct_type: Logical data product type (image etc.). `image`, `cube`,
            `spectrum`, `sed`, `timeseries`, `visibility`, `event` or `measurements`.

        dataproduct_subtype: Data product specific type defined by the ObsTAP provider.
            This is not a useful value for global discovery, but within an archive.

        calib_level: Calibration level {0, 1, 2, 3, 4} (mandatory).
            - 0: Raw instrumental data.
            - 1: Instrumental data in a starndard format (FITS, VOTable, etc.)
            - 2: Calibrated, science ready measurements without instrument signature.
            - 3: Enhanced data products like mosaics, drizzled images or heavily
                processed survey fields. May represent a combination of data from
                multiple primary obs.
            - 4: Analysis data products generated after scientific data manipulation.

        obs_collection: Name of the data collection (mandatory). Either registered
            shortname, full registered IVOA identifier or a data provider defined
            shortname. Often used pattern: `<facility-name>/<instrument-name>`.

        obs_id: Observation ID (mandatory).
            All data-products from a single observation should share the same `obs_id`.
            This is just a unique str-ID with no form. Must be unique to a provider.

        obs_publisher_did: Dataset identifier given by the publisher (mandatory).
            IVOA dataset identifier. Must be a unique value within the namespace
            controlled by the dataset publisher (data center).

        obs_title: Brief description of dataset in free format.

        obs_creator_did: IVOA dataset identifier given by the creator.

        target_class: Class of the target/object as in SSA.
            Either SIMBAD-DB (see https://simbad.cds.unistra.fr/guide/otypes.htx
            `Object type code`), OR NED-DB types
            (see https://ned.ipac.caltech.edu/help/ui/nearposn-list_objecttypes).

        access_url: URL used to access (download) dataset.

        access_format: File content format (MIME type) (`fits`, `jpeg`, `zip`, etc.).

        access_estsize: [kbyte] Estimated size of dataset from `access_url`.

        target_name: Astronomical object observed, if any. This is typically the name
            of an astronomical object, but could be the name of a survey field.

        s_ra: [deg] Central right ascension, ICRS.

        s_dec: [deg] Central declination, ICRS.

        s_fov: [deg] Region covered by the data product. For a circular region, this
            is the diameter. For most data products, the value should be large enough
            to include the entire are of the observation. For detailed spatial
            coverage, the `s_region` attribute can be used.

        s_region: Sky region covered by the data product (expressed in ICRS frame).
            It's a 'spoly` (spherical polygon) type, which is described in
            `https://pgsphere.github.io/doc/funcs.html#funcs.spoly`. Use `spoly`,
            `scircle` or `spoint` to create the formatted str. E.g. for spoly:
            `{(204.712d,+47.405d),(204.380d,+48.311d),(202.349d,+49.116d),
            (200.344d,+48.458d),(199.878d,+47.521d),(200.766d,+46.230d),
            (202.537d,+45.844d),(204.237d,+46.55d)}`.

        s_resolution: [arcsec] Smallest resolvable spatial resolution of data as FWHM.
            If spatial frequency sampling is complex (e.g. interferometry), a typical
            value for spatial resolution estimate should be given.

        s_xel1: Number of elements along the first spatial axis.

        s_xel2: Number of elements along the second spatial axis.

        s_pixel_scale: Sampling period in world coordinate units along the spatial axis.
            It's the distance in WCS units between two pixel centers.

        t_min: [d] Observation start time in Modified Julian Day (MJD).

        t_max: [d] Observation stop time in Modified Julian Day (MJD).

        t_exptime: [s] Total exposure time. For simple exposures: `t_max` - `t_min`.
            For data where the exposure is not constant over the entire data product,
            the median exposure time per pixel is a good way to characterize the typical
            value.

        t_resolution: [s] Minimal interpretable interval between two points along time.
            This can be an average or representative value. For products with no
            sampling along the time axis, it could be set to exposure time or null.

        t_xel: Number of elements along the time axis.

        em_min: [m] Minimal spectral value observed, expressed as a vacuum wavelength.

        em_max: [m] Maximum spectral value observed, expressed as a vacuum wavelength.

        em_res_power: Spectral resolving power `λ / δλ`.

        em_xel: Number of elements along the spectral axis.

        em_ucd: Nature of the spectral axis. This is an em (electromagnetic spectrum)
            UCD (UCD-string see `o_ucd`), e.g. `em.freq`, `em.wl` or `em.energy`.
            Note: For ObsTAP implementation, the spectral axis coordinates are
            constrained as a wavelength quantity expressed in meters.

        o_ucd: UCD (semantic annotation(s)) of observable (e.g. phot.flux.density).
            A UCD is a string containing `;` separated words, which can be separated
            into atoms. The UCD-list is evolving over time and far too extensive to
            describe here. Please have a look at the recommended version of IVOA
            `UCDlist` document at `https://www.ivoa.net/documents/index.html`.

        pol_states: List of polarization states or NULL if not applicable.
            Allowed: I, Q, U, V, RR, LL, RL, LR, XX, YY, XY, YX, POLI, POLA.

        pol_xel: Number of polarization samples in `pol_states`.

        facility_name: Name of the facility used for this observation.

        instrument_name: Name of the instrument used for this observation.
    """

    dataproduct_type: Optional[_DataProductTypeType] = None
    dataproduct_subtype: Optional[str] = None
    calib_level: Optional[_CalibLevelType] = None  # mandatory field
    obs_collection: Optional[str] = None  # mandatory field
    obs_id: Optional[str] = None  # mandatory field
    obs_publisher_did: Optional[str] = None  # mandatory field
    obs_title: Optional[str] = None
    obs_creator_did: Optional[str] = None
    target_class: Optional[str] = None
    access_url: Optional[str] = None
    access_format: Optional[str] = None
    access_estsize: Optional[int] = None
    target_name: Optional[str] = None
    s_ra: Optional[float] = None
    s_dec: Optional[float] = None
    s_fov: Optional[float] = None
    s_region: Optional[str] = None
    s_resolution: Optional[float] = None
    s_xel1: Optional[int] = None
    s_xel2: Optional[int] = None
    s_pixel_scale: Optional[float] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    t_exptime: Optional[float] = None
    t_resolution: Optional[float] = None
    t_xel: Optional[int] = None
    em_min: Optional[float] = None
    em_max: Optional[float] = None
    em_res_power: Optional[float] = None
    em_xel: Optional[int] = None
    em_ucd: Optional[str] = None
    o_ucd: Optional[str] = None
    pol_states: Optional[str] = None
    pol_xel: Optional[int] = None
    facility_name: Optional[str] = None
    instrument_name: Optional[str] = None

    def to_json(
        self,
        fpath: FilePathType,
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
        self.check_ObsCoreMeta(verbose=True)
        mandatory_fields = self._get_mandatory_fields()
        dictionary = asdict(self)
        if ignore_none:
            dictionary = {
                key: value
                for key, value in dictionary.items()
                if value is not None or key in mandatory_fields
            }
        with open(file=fpath, mode="w") as json_file:
            json_file.write(json.dumps(dictionary))

    def set_pol_states(self, pol_states: _PolStatesListType) -> None:
        """Sets `pol_states` from a pythonic interface to a `str` according to ObsCore.

        Overwrites if `pol_states` already exists.

        Args:
            pol_states: Polarisation states.
        """
        all_pol_states = set(get_args(_PolStatesType))
        pol_states_ordered = all_pol_states - (all_pol_states - set(pol_states))
        pol_states_str = "/".join(("", *pol_states_ordered, ""))
        self.pol_states = pol_states_str

    def get_pol_states(self) -> Optional[_PolStatesListType]:
        """Parses the polarisation states to `_PolStatesListType`.

        Returns:
            List of polarisation states if field-value is not None.
        """

        def check_pol_type(pol_states: List[str]) -> TypeGuard[_PolStatesListType]:
            valid_pol_states = get_args(_PolStatesType)
            return all(pol_state in valid_pol_states for pol_state in pol_states)

        if self.pol_states is None:
            return None
        pol_states = self.pol_states.split("/")[1:-1]
        if check_pol_type(pol_states=pol_states):
            return pol_states
        else:
            err_msg = f"Invalid polarization values encountered in {self.pol_states=}"
            raise ValueError(err_msg)

    def check_ObsCoreMeta(self, verbose: bool = False) -> bool:
        """Checks whether `ObsCoreMeta` is ready for serialization.

        This doesn't perform a full check if all field-values are valid.
        Currently supported checks:
        - Presence of mandatory fields
        - Polarization fields
        - Axes fields

        Args:
            verbose: Verbose?

        Returns:
            True if ready, else False.
        """
        return (
            self._check_mandatory_fields(
                verbose=verbose,
            )
            and self._check_polarization(
                verbose=verbose,
            )
            and self._check_axes(
                verbose=verbose,
            )
        )

    @overload
    def from_visibility(
        self,
        vis: Visibility,
        calibrated: Optional[bool],
        inode: Optional[Union[str, Path]],
        tel: Telescope,
        obs: Optional[Observation],
    ) -> Dict[str, Any]:
        ...

    @overload
    def from_visibility(
        self,
        vis: Visibility,
        calibrated: Optional[bool],
        inode: Optional[Union[str, Path]],
        tel: Literal[None],
        obs: Literal[None],
    ) -> Dict[str, Any]:
        ...

    def from_visibility(
        self,
        vis: Visibility,
        calibrated: Optional[bool] = None,
        inode: Optional[Union[str, Path]] = None,
        tel: Optional[Telescope] = None,
        obs: Optional[Observation] = None,
    ) -> Dict[str, Any]:  # change to get actual dataclass?
        """Suggests fields from `Visibility`.

        Supported visibility formats: `OSKAR .vis` files.

        Not supported atm: `CASA .ms` measurement sets.

        Args:
            vis: `Visibility` instance.
            calibrated: Calibrated visibilities?
            inode: Estimate size of `inode`? Can take a while for very large dirs.
            tel: `Telescope` to determine smallest spatial resolution.
            obs: `Observation` to determine smallest spatial resolution.
        """
        out: Dict[str, Any] = {"dataproduct_type": "visibility"}
        # only .vis supported atm, Oskar can't read .ms with multiple spectral windows
        if os.path.exists(vis.vis_path):  # assumes `vis_path` tp be
            header, _ = VisHeader.read(vis.vis_path)
            out["s_ra"] = header.phase_centre_ra_deg
            out["s_dec"] = header.phase_centre_dec_deg
            time_start_mjd_utc = header.time_start_mjd_utc
            out["t_min"] = time_start_mjd_utc
            time_inc_sec = header.time_inc_sec
            total_duration_sec = time_inc_sec * header.num_times_total
            time_end_mjd_utc = time_start_mjd_utc + (total_duration_sec / 86400.0)
            out["t_max"] = time_end_mjd_utc
            out["t_exptime"] = (
                time_end_mjd_utc - time_start_mjd_utc
            )  # assumes constant exposure time
            t_res = max(time_inc_sec, header.time_average_sec)
            out["t_resolution"] = t_res
            num_elements_t = int(total_duration_sec / t_res)
            out["t_xel"] = num_elements_t
            freq_start_hz = header.freq_start_hz  # midpoint freq of first channel
            channel_bandwidth_hz = header.channel_bandwidth_hz
            freq_inc_hz = header.freq_inc_hz
            min_freq_hz = freq_start_hz - channel_bandwidth_hz / 2
            num_channels_total = header.num_channels_total
            max_freq_hz = min_freq_hz + freq_inc_hz * num_channels_total
            c = 299792458  # m/s
            min_wavelength_m = c / max_freq_hz
            out["em_min"] = min_wavelength_m
            max_wavelength_m = c / freq_start_hz
            out["em_max"] = max_wavelength_m
            if freq_inc_hz != 0:
                midpoint_frequency_hz = (freq_start_hz + max_freq_hz) / 2
                out["em_res_power"] = midpoint_frequency_hz / freq_inc_hz
            out["em_xel"] = num_channels_total
        out["em_ucd"] = "em.energy;em.radio"
        out["o_ucd"] = "phot.flux.density;phys.polarization.stokes"
        # no particular polarization infos here for `pol_states` & `pol_xel`
        # need dish/antenna size for `s_fov` & `s_region` (tracking-mode)
        if calibrated is not None:
            if calibrated:
                out["calib_level"] = 2
            else:
                out["calib_level"] = 1
        if inode is not None:  # supports any format (e.g. `.vis`, `.ms`, `.fits`, ...)
            out["access_estsize"] = int(getsize(inode=inode) / 1e3)  # B -> KB
            # as far as I know, there's no MIME type for .ms or .vis, but for `.fits`
        if tel is not None and (tel_name := tel.name) is not None:
            out["instrument_name"] = tel_name
        if tel is not None and obs is not None:
            freq_inc_hz = obs.frequency_increment_hz
            min_freq_hz = obs.start_frequency_hz - freq_inc_hz / 2
            end_freq_hz = min_freq_hz + freq_inc_hz * obs.number_of_channels
            b = float(tel.longest_baseline())
            out["s_resolution"] = tel.ang_res(freq=end_freq_hz, b=b)
        return out

    def from_image(
        self,
        img: Image,
        inode: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:  # change to get actual dataclass?
        """Update fields from `Image`.

        Note: Because of potential inconsistencies between different .fits images
            regarding header-keywords and their usage, only a minimal set of keywords
            are considered with the following assumptions:

        - `CDELT1` & `CDELT2`, (X,Y) coordinate increment at reference point are in deg.
        - `NAXIS1` & `NAXIS2`, (X,Y) resolution of image.
        - `CRVAL1` & `CRVAL2`, (RA,DEC) are in deg.

        If the keywords considered made are not sufficient, extend the result yourself.
        In case the assumptions are wrong for your use-case, don't use this function.

        Args:
            img: `Image` instance.
            inode: Estimate size of `inode`?
        """
        assert_valid_ending(path=img.path, ending=".fits")
        out: Dict[str, Any] = {
            "dataproduct_type": "image",
            "calib_level": 3,  # I think images are always a 3?
            "access_format": "image/fits",
        }
        header = img.header
        if not header["SIMPLE"]:
            wmsg = f"{img.path} doesn't follow fits standard!"
            warn(message=wmsg, category=UserWarning, stacklevel=1)
        x_inc_deg = img.header["CDELT1"]
        y_inc_deg = img.header["CDELT2"]
        x_pixel = img.header["NAXIS1"]
        y_pixel = img.header["NAXIS2"]
        ra_deg = img.header["CRVAL1"]
        dec_deg = img.header["CRVAL2"]

        fov_deg = np.sqrt(  # noqa: F841
            (x_inc_deg * x_pixel) ** 2 + (y_inc_deg * y_pixel) ** 2
        )  # circular fov (flattened)
        half_width_deg = abs(x_inc_deg) * x_pixel / 2
        half_height_deg = abs(y_inc_deg) * y_pixel / 2
        bottom_left = (ra_deg - half_width_deg, dec_deg - half_height_deg)  # noqa: F841
        top_left = (ra_deg - half_width_deg, dec_deg + half_height_deg)  # noqa: F841
        top_right = (ra_deg + half_width_deg, dec_deg + half_height_deg)  # noqa: F841
        bottom_right = (  # noqa: F841
            ra_deg + half_width_deg,
            dec_deg - half_height_deg,
        )

        # TODO: `s_ra`, `s_dec`, `s_fov`, `s_region`, `s_resolution`, `s_xel1`, `s_xel2`
        # TODO: `s_pixel_scale`, `em_xel`, `em_min`?, `em_max`?, `em_res_power`?
        # TODO: `em_ucd`?, `o_ucd`?, `pol_states`?, `pol_xel`?
        if inode is not None:
            out["access_estsize"] = int(getsize(inode=inode) / 1e3)  # B -> KB
        return out

    def set_fields(self, **kwargs: Any) -> None:
        """Set fields from `kwargs` if they're valid.

        Args:
            kwargs: Field names and values to set.
        """
        field_names = [field.name for field in fields(self)]
        for k, v in kwargs.items():
            if k not in field_names:
                wmsg = (
                    f"Skipping `{k}` because it's not a valid field of `ObsCoreMeta`."
                )
                warn(message=wmsg, category=UserWarning, stacklevel=1)
                continue
            setattr(self, k, v)

    @classmethod
    def spoly(
        cls,
        poly: Sequence[tuple[float, float]],
        *,
        ndigits: int = 3,
        suffix: str = "d",
    ) -> str:
        """Converts `poly` to spoly str for `s_region`.

        spoly: https://pgsphere.github.io/doc/funcs.html

        E.g. `{(0,0),(1,0),(1,1)}`

        Args:
            args: Consecutive RA,DEC [deg] poly tuples.
            ndigits: Number of digits to round.
            suffix: Suffix for each number (e.g. "d" for deg).

        Returns:
            spoly str.
        """
        spoly_str = ""
        if len(poly) < 3:
            err_msg = "Polygon must have at least 3 (non-colinear) points!"
            raise RuntimeError(err_msg)
        for point in poly:
            ra_str = cls._convert(
                number=point[0],
                axis="RA",
                ndigits=ndigits,
                suffix=suffix,
            )
            dec_str = cls._convert(
                number=point[1],
                axis="DEC",
                ndigits=ndigits,
                suffix=suffix,
            )
            spoly_str += f"({ra_str},{dec_str}),"
        return "{" + spoly_str[:-1] + "}"

    @classmethod
    def spoint(
        cls,
        point: tuple[float, float],
        *,
        ndigits: int = 3,
        suffix: str = "d",
    ) -> str:
        """Converts `point` to spoint str for `s_region`.

        spoint: https://pgsphere.github.io/doc/funcs.html

        E.g. `(10d,20d)`

        Args:
            point: RA,DEC [deg] point.
            ndigits: Number of digits to round.
            suffix: Suffix for each number (e.g. "d" for deg).

        Returns:
            spoint str.
        """
        ra_str = cls._convert(
            number=point[0],
            axis="RA",
            ndigits=ndigits,
            suffix=suffix,
        )
        dec_str = cls._convert(
            number=point[1],
            axis="DEC",
            ndigits=ndigits,
            suffix=suffix,
        )
        return f"({ra_str},{dec_str})"

    @classmethod
    def scircle(
        cls,
        point: tuple[float, float],
        radius: float,
        *,
        ndigits: int = 3,
        suffix: str = "d",
    ) -> str:
        """Converts `point` & `radius` to scircle str for `s_region`.

        scircle: https://pgsphere.github.io/doc/funcs.html

        E.g. `<(0d,90d),60d>`

        Args:
            point: RA,DEC [deg] point.
            radius: Radius [deg].
            ndigits: Number of digits to round.
            suffix: Suffix for each number (e.g. "d" for deg).

        Returns:
            scircle str.
        """
        ra_str = cls._convert(
            number=point[0],
            axis="RA",
            ndigits=ndigits,
            suffix=suffix,
        )
        dec_str = cls._convert(
            number=point[1],
            axis="DEC",
            ndigits=ndigits,
            suffix=suffix,
        )
        radius_str = cls._convert(
            number=radius,
            axis="radius",
            ndigits=ndigits,
            suffix=suffix,
        )
        return f"<({ra_str},{dec_str}),{radius_str}>"

    @classmethod
    def _convert(
        cls,
        number: float,
        axis: Literal["RA", "DEC", "radius"],
        *,
        ndigits: int,
        suffix: str = "",
    ) -> str:
        """Converts `number` to specified format.

        Conversion according to args. `axis="DEC"` will get a sign attached.

        Args:
            number: Number to convert.
            axis: RA will get coerced to [0,360], DEC checked for [-90,90] &
                radius checked for [0,180]
            ndigits: Number of digits to round.
            suffix: str to attach at the end (e.g. "d").

        Returns:
            Converted `number`.
        """
        if axis == "RA":
            if number > 360.0 or number < 0.0:
                wmsg = f"Coercing {axis}={number} to {axis}={number}%360"
                warn(message=wmsg, category=UserWarning, stacklevel=1)
            number = number % 360
            sign = ""
        elif axis == "DEC":
            if number < -90.0 or number > 90.0:
                err_msg = f"DEC [deg] must be in range [-90,90], but {number=}"
                raise ValueError(err_msg)
            if number < 0.0:
                sign = "-"
            else:
                sign = "+"
        elif axis == "radius":
            if number < 0.0 or number > 180.0:
                err_msg = f"radius [deg] must be in range [0.180], but {number=}"
                raise ValueError(err_msg)
        else:
            assert_never(axis)
        number = round(number=number, ndigits=ndigits)

        return f"{sign}{number}{suffix}"

    @classmethod
    def _get_mandatory_fields(cls) -> Tuple[str, ...]:
        """Gets the mandatory fields according to `REC-ObsCore-v1.1`.

        Returns:
            Mandatory field-names as tuple.
        """
        return (
            "calib_level",
            "obs_collection",
            "obs_id",
            "obs_publisher_did",
        )

    def _check_mandatory_fields(self, verbose: bool) -> bool:
        """Checks mandatory fields.

        Prints a warning to stderr if one or more field-values are None.

        Args:
            verbose: Verbose?

        Returns:
            True if mandatory fields are all set, else False.
        """
        mandatory_fields = self._get_mandatory_fields()
        mandatory_missing = [
            field_name
            for field_name in mandatory_fields
            if getattr(self, field_name) is None
        ]
        valid: bool = True
        if len(mandatory_missing) > 0:
            valid = False
            if verbose:
                wmsg = (
                    f"{mandatory_missing=} fields are None in `ObsCoreMeta`, "
                    + "but are mandatory to ObsTAP services."
                )
                warn(message=wmsg, category=UserWarning, stacklevel=1)
        return valid

    def _check_polarization(self, verbose: bool) -> bool:
        """Checks polarization fields according to `REC-ObsCore-v1.1`.

        Args:
            verbose: Verbose?

        Returns:
            True if polarization check succeeded without any issues, else False.
        """
        pol_xel = self.pol_xel
        pol_states = self.pol_states
        valid: bool = True
        if pol_states is not None:
            try:
                _ = self.get_pol_states()
            except ValueError as ve:
                valid = False
                if verbose:
                    warn(message=str(ve), category=UserWarning, stacklevel=1)
        if pol_xel is None and pol_states is not None:
            valid = False
            if verbose:
                wmsg = f"`pol_xel` should be specified because {pol_states=}"
                warn(message=wmsg, category=UserWarning, stacklevel=1)
        elif pol_xel is not None and pol_states is None:
            valid = False
            if verbose:
                wmsg = (
                    f"{pol_xel=} is specified, but {pol_states=} which isn't consistent"
                )
                warn(message=wmsg, category=UserWarning, stacklevel=1)
        elif pol_xel is not None and pol_states is not None:
            if pol_xel != (num_pol_states := len(pol_states)):
                wmsg = f"{pol_xel=} should be {num_pol_states=}"
        if (pol_xel is not None or pol_states is not None) and (
            self.o_ucd is None or (ucd_str := "phys.polarisation") not in self.o_ucd
        ):
            valid = False
            if verbose:
                wmsg = f"`o_ucd` must at least contain '{ucd_str}' but it doesn't"
                warn(message=wmsg, category=UserWarning, stacklevel=1)
        return valid

    def _check_axes(self, verbose: bool) -> bool:
        """Checks axis-lengths (`s_xel1`, `s_xel2`, `em_xel`, `t_xel`, `pol_xel`).

        Args:
            verbose: Verbose?

        Returns:
            True if checks passed, else False.
        """
        valid = True

        def check_value(value: int | None) -> bool:
            return value is None or value >= -1

        axis_field_names = ("s_xel1", "s_xel2", "em_xel", "t_xel", "pol_xel")
        invalid_fields: Dict[str, str] = {}
        for axis_field_name in axis_field_names:
            field_value = getattr(self, axis_field_name)
            if not check_value(value=field_value):
                invalid_fields[axis_field_name] = field_value
        if len(invalid_fields) > 0:
            valid = False
            if verbose:
                wmsg = f"Invalid axes-values: {invalid_fields}"
                warn(message=wmsg, category=UserWarning, stacklevel=1)
        return valid
