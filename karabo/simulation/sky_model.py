import copy
import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import oskar
import pandas as pd
from astropy import units as u
from astropy import wcs as awcs
from astropy.table import Table
from astropy.visualization.wcsaxes import SphericalCircle

from karabo.data.external_data import GLEAMSurveyDownloadObject, MIGHTEESurveyDownloadObject
from karabo.util.plotting_util import get_slices


class SkyModel:
    """
    Class containing all information of the to be observed Sky.

    :ivar sources:  List of all point sources in the sky.
                    A single point source consists of:

                    - right ascension (deg)
                    - declination (deg)
                    - stokes I Flux (Jy)
                    - stokes Q Flux (Jy): defaults to 0
                    - stokes U Flux (Jy): defaults to 0
                    - stokes V Flux (Jy): defaults to 0
                    - reference_frequency (Hz): defaults to 0
                    - spectral index (N/A): defaults to 0
                    - rotation measure (rad / m^2): defaults to 0
                    - major axis FWHM (arcsec): defaults to 0
                    - minor axis FWHM (arcsec): defaults to 0
                    - position angle (deg): defaults to 0
                    - source id (object): defaults to None

    """

    def __init__(self, sources: np.ndarray = None, wcs: awcs = None):
        """
        Initialization of a new SkyModel

        :param sources: Adds point sources
        :param wcs: world coordinate system
        """
        self.num_sources: int = 0
        self.shape: tuple = (0, 0)
        self.sources: np.ndarray = None
        self.wcs: awcs = wcs
        self.sources_m = 13
        if sources is not None:
            self.add_point_sources(sources)

    def __get_empty_sources(self, n_sources):
        empty_sources = np.hstack(
            (np.zeros((n_sources, self.sources_m - 1)), np.array([[None] * n_sources]).reshape(-1, 1)))
        return empty_sources

    def add_point_sources(self, sources: np.ndarray):
        """
        Add new point sources to the sky model.

        :param sources: Array-like with shape (number of sources, 13). Each row representing one source.
                        The indices in the second dimension of the array correspond to:

                        - [0] right ascension (deg)-
                        - [1] declination (deg)
                        - [2] stokes I Flux (Jy)
                        - [3] stokes Q Flux (Jy): defaults to 0
                        - [4] stokes U Flux (Jy): defaults to 0
                        - [5] stokes V Flux (Jy): defaults to 0
                        - [6] reference_frequency (Hz): defaults to 0
                        - [7] spectral index (N/A): defaults to 0
                        - [8] rotation measure (rad / m^2): defaults to 0
                        - [9] major axis FWHM (arcsec): defaults to 0
                        - [10] minor axis FWHM (arcsec): defaults to 0
                        - [11] position angle (deg): defaults to 0
                        - [12] source id (object): defaults to None

        """
        if len(sources.shape) > 2:
            return
        if 2 < sources.shape[1] < self.sources_m + 1:
            if sources.shape[1] < self.sources_m:
                # if some elements are missing fill them up with zeros except `source_id`
                missing_shape = self.sources_m - sources.shape[1]
                fill = self.__get_empty_sources(sources.shape[0])
                fill[:, :-missing_shape] = sources
                sources = fill
            if self.sources is not None:
                self.sources = np.vstack((self.sources, sources))
            else:
                self.sources = sources
            self.__update_sky_model()

    def add_point_source(self, right_ascension: float, declination: float, stokes_I_flux: float,
                         stokes_Q_flux: float = 0, stokes_U_flux: float = 0, stokes_V_flux: float = 0,
                         reference_frequency: float = 0, spectral_index: float = 0, rotation_measure: float = 0,
                         major_axis_FWHM: float = 0, minor_axis_FWHM: float = 0, position_angle: float = 0,
                         source_id: object = None):
        """
        Add a single new point source to the sky model.

        :param right_ascension:
        :param declination:
        :param stokes_I_flux:
        :param stokes_Q_flux:
        :param stokes_U_flux:
        :param stokes_V_flux:
        :param reference_frequency:
        :param spectral_index:
        :param rotation_measure:
        :param major_axis_FWHM:
        :param minor_axis_FWHM:
        :param position_angle:
        :param source_id:
        """
        new_sources = np.array(
            [[right_ascension, declination, stokes_I_flux, stokes_Q_flux, stokes_U_flux,
              stokes_V_flux, reference_frequency, spectral_index, rotation_measure,
              major_axis_FWHM, minor_axis_FWHM, position_angle, source_id]])
        if self.sources is not None:
            self.sources = np.vstack(self.sources, new_sources)
        else:
            self.sources = new_sources
        self.__update_sky_model()

    def to_array(self, with_obj_ids: bool = False) -> np.ndarray:
        """
        Gets the sources as np.ndarray

        :param with_obj_ids: Option whether object ids should be included or not

        :return: the sources of the SkyModel as np.ndarray
        """
        if with_obj_ids:
            return self[:]
        else:
            return self[:, :-1]

    def filter_by_radius(self, inner_radius_deg: float, outer_radius_deg: float, ra0_deg: float, dec0_deg: float):
        """
        Filters the sky according the an inner and outer circle from the phase center

        :param inner_radius_deg: Inner radius in degrees
        :param outer_radius_deg: Outer raidus in degrees
        :param ra0_deg: Phase center right ascention
        :param dec0_deg: Phase center declination
        :return sky: Filtered copy of the sky
        """
        copied_sky = copy.deepcopy(self)
        inner_circle = SphericalCircle((ra0_deg * u.deg, dec0_deg * u.deg), inner_radius_deg * u.deg)
        outer_circle = SphericalCircle((ra0_deg * u.deg, dec0_deg * u.deg), outer_radius_deg * u.deg)
        outer_sources = outer_circle.contains_points(copied_sky[:, 0:2]).astype('int')
        inner_sources = inner_circle.contains_points(copied_sky[:, 0:2]).astype('int')
        filtered_sources = np.array(outer_sources - inner_sources, dtype='bool')
        filtered_sources_idxs = np.where(filtered_sources == True)[0]
        copied_sky.sources = copied_sky.sources[filtered_sources_idxs]
        copied_sky.__update_sky_model()
        return copied_sky

    def filter_by_flux(self, min_flux_jy: float, max_flux_jy: float):
        """
        Filters the sky using the Stokes-I-flux
        Values outside the range are removed

        :param min_flux_jy: Minimum flux in Jy
        :param max_flux_jy: Maximum flux in Jy
        """
        stokes_I_flux = self[:, 2]
        filtered_sources_idxs = np.where(np.logical_and(stokes_I_flux <= max_flux_jy, stokes_I_flux >= min_flux_jy))[0]
        self.sources = self.sources[filtered_sources_idxs]
        self.__update_sky_model()

    def filter_by_frequency(self, min_freq: float, max_freq: float):
        """
        Filters the sky using the referency frequency in Hz

        :param min_freq: Minimum frequency in Hz
        :param max_freq: Maximum frequency in Hz
        """
        freq = self[:, 6]
        filtered_sources_idxs = np.where(np.logical_and(freq <= max_freq, freq >= min_freq))[0]
        self.sources = self.sources[filtered_sources_idxs]
        self.__update_sky_model()

    def get_wcs(self) -> awcs:
        """
        Gets the currently active world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html

        :return: world coordinate system
        """
        return self.wcs

    def set_wcs(self, wcs: awcs):
        """
        Sets a new world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html

        :param wcs: world coordinate system
        """
        self.wcs = wcs

    def setup_default_wcs(self, phase_center: list = [0, 0]) -> awcs:
        """
        Defines a default world coordinate system astropy.wcs
        For more details see https://docs.astropy.org/en/stable/wcs/index.html

        :param phase_center: ra-dec location

        :return: wcs
        """
        w = awcs.wcs.WCS(naxis=2)
        w.wcs.crpix = [0, 0]  # coordinate reference pixel per axis
        w.wcs.cdelt = [-1, 1]  # coordinate increments on sphere per axis
        w.wcs.crval = phase_center
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]  # coordinate axis type
        self.wcs = w
        return w

    @staticmethod
    def get_fits_catalog(path: str) -> Table:
        """
        Gets astropy.table.table.Table from a .fits catalog

        :param path: Location of the .fits file

        :return: fits catalog
        """
        return Table.read(path)

    def explore_sky(self, phase_center: np.ndarray = np.array([0, 0]), xlim: tuple = (-1, 1), ylim: tuple = (-1, 1),
                    figsize: tuple = (6, 6), title: str = '', xlabel: str = '', ylabel: str = '',
                    s: float = 20, cfun: Callable = np.log10, cmap: str = 'plasma', cbar_label: str = '',
                    with_labels: bool = False, wcs: awcs = None):
        """
        A scatter plot of y vs. x of the point sources of the SkyModel

        :param phase_center:
        :param xlim: limit of plot in degrees from phase centre in x direction
        :param ylim: limit of plot in degrees from phase centre in y direction
        :param figsize: figure size
        :param title: plot titble
        :param xlabel: xlabel override
        :param ylabel: ylabel override
        :param s: size of scatter points
        :param cfun: color function
        :param cmap: color map
        :param cbar_label: color bar label
        :param with_labels: Plots object ID's if set
        :param wcs: If you want to use a custom astropy.wcs, ignores phase_center if set
        """
        if wcs is None:
            wcs = self.setup_default_wcs(phase_center)
        px, py = wcs.wcs_world2pix(self[:, 0], self[:, 1], 1)  # ra-dec transformation

        flux, vmin, vmax = None, None, None
        if cmap is not None and cfun is not None:
            flux = self[:, 2]
            flux = cfun(flux)
            vmin, vmax = np.min(flux), np.max(flux)
        else:  # set both to None if one of them is None
            cfun = None
            cmap = None

        slices = get_slices(wcs)

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=wcs, slices=slices))
        sc = ax.scatter(px, py, s=s, c=flux, cmap=cmap, vmin=vmin, vmax=vmax)

        if with_labels:
            for i, txt in enumerate(self[:, -1]):
                ax.annotate(txt, (px[i], py[i]))

        plt.axis('equal')
        plt.colorbar(sc, label=cbar_label)
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_sky(self, phase_center):
        ra0, dec0 = phase_center[0], phase_center[1]
        data = self[:, 0:3]
        ra = np.radians(data[:, 0] - ra0)
        dec = np.radians(data[:, 1])
        flux = data[:, 2]
        log_flux = np.log10(flux)
        x = np.cos(dec) * np.sin(ra)
        y = np.cos(np.radians(dec0)) * np.sin(dec) - \
            np.sin(np.radians(dec0)) * np.cos(dec) * np.cos(ra)
        sc = plt.scatter(x, y, s=.5, c=log_flux, cmap='plasma',
                         vmin=np.min(log_flux), vmax=np.max(log_flux))
        plt.axis('equal')
        plt.xlabel('x direction cosine')
        plt.ylabel('y direction cosine')
        plt.colorbar(sc, label='Log10(Stokes I flux [Jy])')
        plt.show()

    def get_OSKAR_sky(self) -> oskar.Sky:
        """
        Get OSKAR sky model object from the defined Sky Model

        :return: oskar sky model
        """
        return oskar.Sky.from_array(self[:, :-1])

    def __update_sky_model(self):
        """
        Updates instance variables of the SkyModel
        """
        self.num_sources = self.sources.shape[0]
        self.shape = self.sources.shape

    def __getitem__(self, key):
        """
        Allows to get access to self.sources in an np.ndarray like manner
        If casts the selected array/scalar to float64 if possible (usually if source_id is not selected)

        :param key: slice key

        :return: sliced self.sources
        """
        sources = self.sources[key]
        try:
            sources = np.float64(sources)
        except ValueError:
            pass  # nothing toDo here
        return sources

    def __setitem__(self, key, value):
        """
        Allows to set values in an np.ndarray like manner

        :param key: slice key
        :param value: values to store
        """
        if self.sources is None:
            self.sources = self.__get_empty_sources(len(value))
        self.sources[key] = value

    def save_sky_model_as_csv(self, path: str):
        """
        Save source array into a csv.
        :param path: path to save the csv file in.
        """
        pd.DataFrame(self.sources).to_csv(path, index=False,
                                          header=["right ascension (deg)",
                                                  "declination (deg)",
                                                  "stokes I Flux (Jy)",
                                                  "stokes Q Flux (Jy)",
                                                  "stokes U Flux (Jy)",
                                                  "stokes V Flux (Jy)",
                                                  "reference_frequency (Hz)",
                                                  "spectral index (N/A)",
                                                  "rotation measure (rad / m^2)",
                                                  "major axis FWHM (arcsec)",
                                                  "minor axis FWHM (arcsec)",
                                                  "position angle (deg)",
                                                  "source id (object)"])

    @staticmethod
    def __convert_ra_dec_to_cartesian(ra, dec):
        x = math.cos(ra) * math.cos(dec)
        y = math.sin(ra) * math.cos(dec)
        z = math.sin(dec)
        r = np.array([x, y, z])
        norm = np.linalg.norm(r)
        if norm == 0:
            return r
        return r / norm

    def get_cartesian_sky(self):
        cartesian_sky = np.squeeze(np.apply_along_axis(
            lambda row: [self.__convert_ra_dec_to_cartesian(float(row[0]), float(row[1]))],
            axis=1, arr=self.sources))
        return cartesian_sky

    def project_sky_to_2d_image(self, cell_size: float, pixel_per_side: int):
        from karabo.Imaging.imager import Imager
        imager = Imager(visibility=None, imaging_cellsize=cell_size, imaging_npixel=pixel_per_side)
        coords = imager.sky_sources_to_pixel_coordinates(cell_size, pixel_per_side, self)
        return coords


def get_GLEAM_Sky() -> SkyModel:
    survey = GLEAMSurveyDownloadObject()
    path = survey.get()
    gleam = SkyModel.get_fits_catalog(path)
    df_gleam = gleam.to_pandas()
    ref_freq = 76e6
    df_gleam = df_gleam[~df_gleam['Fp076'].isna()]
    ra, dec, fp = df_gleam['RAJ2000'], df_gleam['DEJ2000'], df_gleam['Fp076']
    sky_array = np.column_stack((ra, dec, fp, np.zeros(ra.shape[0]), np.zeros(ra.shape[0]),
                                 np.zeros(ra.shape[0]), [ref_freq] * ra.shape[0])).astype('float64')
    sky = SkyModel(sky_array)
    # major axis FWHM, minor axis FWHM, position angle, object id
    sky[:, [9, 10, 11, 12]] = df_gleam[['a076', 'b076', 'pa076', 'GLEAM']]
    return sky

def get_MIGHTEE_Sky() -> SkyModel:
    survey = MIGHTEESurveyDownloadObject()
    path = survey.get()
    mightee = SkyModel.get_fits_catalog(path)
    df_mightee = mightee.to_pandas()
    ref_freq = 76e6
    ra, dec, fp = df_mightee['RA'], df_mightee['DEC'], df_mightee['NU_EFF']
    sky_array = np.column_stack((ra, dec, fp, np.zeros(ra.shape[0]), np.zeros(ra.shape[0]),
                                 np.zeros(ra.shape[0]), [ref_freq] * ra.shape[0])).astype('float64')
    sky = SkyModel(sky_array)
    # major axis FWHM, minor axis FWHM, position angle, object id
    sky[:, [9, 10, 11, 12]] = df_mightee[['IM_MAJ', 'IM_MIN', 'IM_PA', 'NAME']]
    return sky


def read_sky_model_from_csv(path: str) -> SkyModel:
    """
    Read a CSV file in to create a SkyModel.
    The CSV should have the following columns

    - right ascension (deg)
    - declination (deg)
    - stokes I Flux (Jy)
    - stokes Q Flux (Jy): if no information available, set to 0
    - stokes U Flux (Jy): if no information available, set to 0
    - stokes V Flux (Jy): if no information available, set to 0
    - reference_frequency (Hz): if no information available, set to 0
    - spectral index (N/A): if no information available, set to 0
    - rotation measure (rad / m^2): if no information available, set to 0
    - major axis FWHM (arcsec): if no information available, set to 0
    - minor axis FWHM (arcsec): if no information available, set to 0
    - position angle (deg): if no information available, set to 0
    - source id (object): if no information available, set to None

    :param path: file to read in
    :return: SkyModel
    """
    # TODO: add validation of csv
    dataframe = pd.read_csv(path)
    sources = dataframe.to_numpy()
    sky = SkyModel(sources)
    return sky
