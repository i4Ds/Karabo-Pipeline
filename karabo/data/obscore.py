"""ObsCore Data Model.

https://ivoa.net/documents/ObsCore/

Recommended IVOA documents: https://www.ivoa.net/documents/index.html
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
)
from warnings import warn

import numpy as np
import rfc3986
import rfc3986.validators
from astropy import constants as const
from astropy import units as u
from astropy.io.fits.header import Header
from astropy.units.core import UnitBase
from astropy.units.quantity import Quantity
from typing_extensions import Self, TypeGuard, assert_never

from karabo.data.casa import MSMainTable, MSMeta, MSPolarizationTable
from karabo.imaging.image import Image
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


@dataclass(frozen=True)
class FitsHeaderAxis:
    """Fits header axis dataclass.

    Descriptive dataclass for .fits axis and unit allocation infos.

    Args:
        axis: Axis number.
        unit: Unit of value `CRVAL` and increment `CDELT` of axis in the .fits file.
    """

    axis: int
    unit: UnitBase

    def crval(self, header: Header) -> Quantity:
        """CRVAL{`axis`} value at reference point in `CTYPE{`axis`} unit.

        Args:
            header: Header to extract value from.

        Returns:
            Value as astropy `Quantity`.
        """
        return header[f"CRVAL{self.axis}"] * self.unit

    def cdelt(self, header: Header) -> Quantity:
        """CDELT{`axis`} increment at reference point in `CTYPE{`axis`} unit.

        Args:
            header: Header to extract increment from.

        Returns:
            Value as astropy `Quantity`.
        """
        return header[f"CDELT{self.axis}"] * self.unit

    def crpix(self, header: Header) -> float:
        """CRPIX{`axis`} location at reference point along axis.

        Args:
            header: Header to extract location from.

        Returns:
            Location of axis.
        """
        return float(header[f"CRPIX{self.axis}"])

    def naxis(self, header: Header) -> int:
        """NAXIS{`axis`} length of axis.

        Args:
            header: Header to extract length from.

        Returns:
            Length of axis.
        """
        return int(header[f"NAXIS{self.axis}"])

    def ctype(self, header: Header) -> str:
        """CTYPE{`axis`} unit type of axis.

        This is just a str, not an astropy unit-name.

        Args:
            header: Header to extract unit from.

        Returns:
            Unit as str.
        """
        return str(header[f"CTYPE{self.axis}"])


@dataclass(frozen=True)
class FitsHeaderAxes:
    """Fits file axes description.

    Needed for file-parsing for axis-position and unit-transformation.

    Args:
        x: X/RA axis (default: axis=1, unit=deg) of image.
        y: Y/DEC axis (default: axis=2, unit=deg) of image.
        freq: Freq axis (default: axis=3, unit=Hz) of image.
    """

    x: FitsHeaderAxis = field(
        default_factory=lambda: FitsHeaderAxis(axis=1, unit=u.deg)
    )
    y: FitsHeaderAxis = field(
        default_factory=lambda: FitsHeaderAxis(axis=2, unit=u.deg)
    )
    freq: FitsHeaderAxis = field(
        default_factory=lambda: FitsHeaderAxis(axis=3, unit=u.Hz)
    )


@dataclass  # TODO: once when just Python >= 3.10 is supported, change to (kw_only=True)
class ObsCoreMeta:
    r"""IVOA ObsCore v1.1 metadata (TAP column names).

    This doesn't describe a full ObsCoreDM, but the mandatory and the non-mandatory
    fields defined in the SRCNet Rucio database. The actual JSON to send to a specific
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
            - 1: Instrumental data in a standard format (FITS, VOTable, etc.)
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
            controlled by the dataset publisher (data center). `ObsCoreMeta.get_ivoid`
            may help creating this value.

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
            to include the entire area of the observation. For detailed spatial
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

    def to_dict(
        self,
        fpath: Optional[FilePathType] = None,
        *,
        ignore_none: bool = True,
    ) -> Dict[str, Any]:
        """Converts this dataclass into a dict.

        Args:
            fpath: File-path to write dump.
            ignore_none: Ignore non-mandatory `None` fields?

        Returns:
            Dataclass as dict.
        """
        self.check_ObsCoreMeta(verbose=True)
        mandatory_fields = self._get_mandatory_fields()
        dictionary = asdict(self)
        if ignore_none:
            dictionary = {
                key: value
                for key, value in dictionary.items()
                if value is not None or key in mandatory_fields
            }
        if fpath is not None:
            dump = json.dumps(dictionary)
            with open(file=fpath, mode="w") as json_file:
                json_file.write(dump)
        return dictionary

    def set_pol_states(
        self, pol_states: Optional[Union[_PolStatesListType, List[int]]]
    ) -> None:
        """Sets `pol_states` from a pythonic interface to a `str` according to ObsCore.

        Overwrites if `pol_states` already exists.

        Args:
            pol_states: Polarization states.
        """

        def is_int_list(
            val: Union[_PolStatesListType, List[int]],
        ) -> TypeGuard[List[int]]:
            return all(isinstance(x, int) for x in val)

        def is_str_list(
            val: Union[_PolStatesListType, List[int]],
        ) -> TypeGuard[_PolStatesListType]:
            return all(isinstance(x, str) for x in val)

        if pol_states is None or len(pol_states) == 0:
            self.pol_states = None
        else:
            all_pol_states: _PolStatesListType = list(get_args(_PolStatesType))
            if is_int_list(pol_states):
                stokes_types = MSPolarizationTable.get_stokes_type(corr_type=pol_states)
                pol_states_list = cast(
                    _PolStatesListType,
                    [
                        stokes_type
                        for stokes_type in stokes_types
                        if stokes_type in all_pol_states
                    ],
                )
            elif is_str_list(pol_states):
                pol_states_list = pol_states
            else:
                err_msg = f"{pol_states=} must be of type `list[int]` or `list[str]`."
                raise TypeError(err_msg)

            pol_states_ordered = set(all_pol_states) - (
                set(all_pol_states) - set(pol_states_list)
            )
            pol_states_str = "/".join(("", *pol_states_ordered, ""))
            self.pol_states = pol_states_str

    def get_pol_states(self) -> Optional[_PolStatesListType]:
        """Parses the polarization states to `_PolStatesListType`.

        Returns:
            List of polarization states if field-value is not None.
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

    def check_ObsCoreMeta(
        self,
        *,
        verbose: bool = False,
    ) -> bool:
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

    @classmethod
    def from_visibility(
        cls,
        vis: Visibility,  # not sure if also MS-path should be allowed
        *,
        calibrated: Optional[bool] = None,
    ) -> Self:
        """Suggests fields from `Visibility`.

        This function may not adjust each field for your needs. In addition, there
        is no possibility to fill all mandatory fields because there is just no
        information available. Thus, you have to take care of some fields by yourself.

        Supported formats: CASA Measurement Sets.

        Args:
            vis: `Visibility` instance.
            calibrated: Calibrated visibilities?

        Returns:
            `ObsCoreMeta` instance.
        """
        ocm = cls(dataproduct_type="visibility")
        ms_path = vis.ms_file_path
        c = const.c.value
        ms_meta = MSMeta.from_ms(ms_path=ms_path)
        field_table = ms_meta.field
        if not np.all(field_table.num_poly == 0):
            err_msg = (
                "Phase-center(s) depicted as polynomials in time is not supported."
            )
            raise NotImplementedError(err_msg)
        if (n_phase_center := field_table.phase_dir.shape[0]) > 1:
            wmsg = (
                f"Found {n_phase_center=} in `ObsCoreMeta.from_visibility` for "
                + f"{ms_path=}. Computing mean/central phase-center!"
            )
            warn(wmsg, category=RuntimeWarning, stacklevel=2)
        phase_dir_deg = np.rad2deg(field_table.phase_dir)
        phase_dir_deg[:, :, 0] = phase_dir_deg[:, :, 0] % 360  # ensures positive RA
        phase_center_deg = phase_dir_deg.mean(axis=(0, 1))
        ocm.s_ra = phase_center_deg[0]
        ocm.s_dec = phase_center_deg[1]
        observation = ms_meta.observation
        time_start_mjd_utc = observation.time_range[:, 0].min() / 86400
        ocm.t_min = time_start_mjd_utc
        time_end_mjd_utc = observation.time_range[:, 1].max() / 86400
        ocm.t_max = time_end_mjd_utc
        ocm.t_exptime = np.abs(  # abs is just to be sure
            observation.time_range[:, 1] - observation.time_range[:, 0]
        ).mean()  # in s, not mjd-utc
        ocm.t_resolution = np.median(  # chosen representative value over min-interval
            MSMainTable.get_col(
                ms_path=ms_path,
                col="INTERVAL",  # according to MS v2.0
                nrow=300000000,  # to avoid memory & time issues, should be repres
            )
        )
        ocm.t_xel = MSMainTable.nrows(ms_path=ms_path)
        spectral_window = ms_meta.spectral_window
        ocm.em_min = c / np.min(
            spectral_window.chan_freq - spectral_window.chan_width / 2
        )
        ocm.em_max = c / np.max(
            spectral_window.chan_freq + spectral_window.chan_width / 2
        )
        ocm.em_res_power = np.median(
            spectral_window.chan_freq / spectral_window.chan_width
        )
        ocm.em_xel = spectral_window.chan_freq.ravel().shape[0]
        ocm.access_estsize = int(getsize(inode=ms_path) / 1e3)  # B -> KB
        ocm.instrument_name = ",".join(np.unique(ms_meta.observation.telescope_name))
        b = float(np.max(ms_meta.antenna.baseline_dists()))
        ocm.s_resolution = Telescope.ang_res(
            freq=float(np.max(spectral_window.ref_frequency)),
            b=b,
        )
        ocm.pol_xel = MSPolarizationTable.nrows(ms_path=ms_path)
        corr_types = np.unique(ms_meta.polarization.corr_type.ravel()).tolist()
        ocm.set_pol_states(pol_states=corr_types)
        ocm.em_ucd = "em.energy;em.radio"
        ocm.o_ucd = "phot.flux.density;phys.polarisation"
        if calibrated is not None:
            if calibrated:  # can't be extracted from visibilities as far as I know
                ocm.calib_level = 2
            else:
                ocm.calib_level = 1
        # as far as I know, there's no MIME type for .ms or .vis, but for `.fits`
        return ocm

    @classmethod
    def from_image(
        cls,
        img: Image,  # not sure if also fits-path should be allowed
        *,
        fits_axes: Optional[FitsHeaderAxes] = None,
    ) -> Self:
        """Update fields from `Image`.

        This function may not adjust each field for your needs. In addition, there
            is no possibility to fill some mandatory fields because there is just no
            information available. Thus, you have to take care of some fields by
            yourself.

        Note: This function assumes the presence of `NAXIS`, `CRPIX`, `CRVAL` & `CDELT`
            for each axis-number specified in `fits_axes` to be present and correctly
            specified for your .fits file in `img`. Otherwise, this function might fail
            or produce corrupt values.

        Args:
            img: `Image` instance.
            fits_axes: `FitsAxes` instance to specify axis-number and according unit
                to extract and transform information from the .fits file of
                `img` correctly.

        Returns:
            `ObsCoreMeta` instance.
        """
        file = img.path
        assert_valid_ending(path=file, ending=".fits")
        if fits_axes is None:
            fits_axes = FitsHeaderAxes()
        header = img.header
        if not header["SIMPLE"]:
            wmsg = f"{file} doesn't follow .fits standard! Info extraction might fail!"
            warn(message=wmsg, category=UserWarning, stacklevel=2)
        freq_center_hz = fits_axes.freq.crval(header=header).to(u.Hz).value  # center
        x_inc_deg = fits_axes.x.cdelt(header=header).to(u.deg).value  # inc at center
        y_inc_deg = fits_axes.y.cdelt(header=header).to(u.deg).value  # inc at center
        freq_inc_hz = fits_axes.freq.cdelt(header=header).to(u.Hz).value
        x_pixel = fits_axes.x.naxis(header=header)
        y_pixel = fits_axes.y.naxis(header=header)
        n_channels = fits_axes.freq.naxis(header=header)

        if (s_pixel_scale := abs(x_inc_deg)) != abs(y_inc_deg):
            wmsg = (
                f"Pixel-size is not square for `s_pixel_scale`: {x_inc_deg=}, "
                + f"{y_inc_deg}. `s_pixel_scale` set to {s_pixel_scale=}."
            )
            warn(message=wmsg, category=UserWarning, stacklevel=2)
        min_freq_hz = freq_center_hz - freq_inc_hz / 2
        c = const.c.value
        min_wavelength_m = c / min_freq_hz
        max_freq_hz = min_freq_hz + freq_inc_hz * n_channels
        max_wavelength_m = c / max_freq_hz
        fov_deg = np.sqrt(
            (s_pixel_scale * x_pixel) ** 2 + (y_inc_deg * y_pixel) ** 2
        )  # circular fov of flattened image
        world_coords = Image.get_corners_in_world(header=header)
        # assuming `get_corners_in_world` has circling corner order
        corners = [(world_coord[0], world_coord[1]) for world_coord in world_coords]
        spoly = cls.spoly(
            poly=corners,
            ndigits=3,
            suffix="d",
        )
        img_size_kb = int(getsize(inode=file) / 1e3)  # B -> KB
        ocm = cls(
            dataproduct_type="image",
            calib_level=3,  # Are images are always a 3?
            access_format="image/fits",
            s_xel1=x_pixel,
            s_xel2=y_pixel,
            s_pixel_scale=s_pixel_scale,
            s_fov=fov_deg,
            s_region=spoly,
            em_min=min_wavelength_m,
            em_max=max_wavelength_m,
            em_xel=n_channels,
            em_ucd="em.freq;em.radio",
            access_estsize=img_size_kb,
        )
        if n_channels > 1 and freq_inc_hz != 0.0:
            em_res_power = freq_center_hz / freq_inc_hz
            ocm.em_res_power = em_res_power
        return ocm

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
    def get_ivoid(
        cls,
        *,
        authority: str,
        path: Optional[str],
        query: Optional[str],
        fragment: Optional[str],
    ) -> str:
        """Gets the IVOA identifier for `ObsCoreMeta.obs_creator_did`.

        IVOID according to IVOA 'REC-Identifiers-2.0'. Do NOT specify RFC 3986
            delimiters in the input-args, they're added automatically.

        Please set up an Issue if this is not up-to-date anymore.

        Args:
            authority: Organization (usually a data provider) that has been granted
                the right by the IVOA to create IVOA-compliant identifiers for
                resources it registers.
            path: Resource key. It's 'a resource that is unique within the namespace
                of an authority identifier.
            query: According to RFC 3986.
            fragment: According to RFC 3986.

        Returns:
            IVOID.
        """
        # all checks here are IVOID specs on top of RFC 3986
        # not sure if it matters if own or RFC 3986 validation comes first
        if len(authority) < 3:
            err_msg = f"{authority=} must be at least 3 chars long."
            raise ValueError(err_msg)
        if not authority[0].isalnum():
            err_msg = f"{authority=} must begin with an alphanumeric char."
            raise ValueError(err_msg)
        if not cls._is_unreserved(string=authority):
            err_msg = f"{authority=} must only contain unreserved chars, ~ discouraged."
            raise ValueError(err_msg)
        if path is not None and len(path) > 0:
            if path[0] != "/":
                err_msg = f"Non-empty {path=} must start with a `/`."
                raise ValueError(err_msg)
            if cls._contains_percent_encoded_chars(string=path):
                err_msg = f"{path=} must not contain percent-encoded chars."
                raise ValueError(err_msg)
            if cls._contains_col_or_commercial_at_signs(string=path):
                err_msg = f"{path=} contains `:` or `@`."
                raise ValueError(err_msg)
        uri_ref = rfc3986.URIReference(
            scheme="ivo",  # fixed in Identifiers-2.0 spec
            authority=authority,
            path=path,
            query=query,
            fragment=fragment,
        )
        validator = rfc3986.validators.Validator().check_validity_of(
            "scheme", "host", "port", "userinfo", "path", "query", "fragment"
        )
        validator.validate(uri=uri_ref)
        return str(uri_ref.unsplit())

    @classmethod
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
        elif axis == "DEC":
            if number < -90.0 or number > 90.0:
                err_msg = f"DEC [deg] must be in range [-90,90], but {number=}"
                raise ValueError(err_msg)
        elif axis == "radius":
            if number < 0.0 or number > 180.0:
                err_msg = f"radius [deg] must be in range [0.180], but {number=}"
                raise ValueError(err_msg)
        else:
            assert_never(axis)
        number = round(number=number, ndigits=ndigits)

        return f"{number}{suffix}"

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

    def _check_mandatory_fields(self, *, verbose: bool) -> bool:
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

    def _check_polarization(self, *, verbose: bool) -> bool:
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
                valid = False
                if verbose:
                    wmsg = f"{pol_xel=} should be {num_pol_states=}"
                    warn(message=wmsg, category=UserWarning, stacklevel=1)
        if (pol_xel is not None or pol_states is not None) and (
            self.o_ucd is None or (ucd_str := "phys.polarisation") not in self.o_ucd
        ):
            valid = False
            if verbose:
                wmsg = f"`o_ucd` must at least contain '{ucd_str}' but it doesn't"
                warn(message=wmsg, category=UserWarning, stacklevel=1)
        return valid

    def _check_axes(self, *, verbose: bool) -> bool:
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

    @classmethod
    def _is_unreserved(cls, string: str) -> bool:
        """Checks if `string` contains only RFC 3986 unreserved chars.

        Args:
            string: String to check.

        Returns:
            True if `string` contains only unreserved chars, else False.
        """
        unreserved_chars = re.compile(r"^[A-Za-z0-9\-._~]+$")
        return unreserved_chars.match(string) is not None

    @classmethod
    def _contains_percent_encoded_chars(cls, string: str) -> bool:
        """Checks if `string` contains percent-encoded chars.

        This is a % followed by exactly two hexadecimal digits.

        Args:
            string: String to check.

        Returns:
            True if `string` contains percent-encoded chars, else False.
        """
        percent_encoded_pattern = re.compile(r"%[0-9A-Fa-f]{2}")
        return percent_encoded_pattern.search(string) is not None

    @classmethod
    def _contains_col_or_commercial_at_signs(cls, string: str) -> bool:
        """Checks for colons and commercial at signs in `string`.

        Args:
            string: String to check.

        Returns:
            True if `string` contains one of the chars, else False.
        """
        return ":" in string or "@" in string
