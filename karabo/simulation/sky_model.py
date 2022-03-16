from os import stat
from re import A
from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import oskar
from astropy.table import Table
from astropy.visualization.wcsaxes import SphericalCircle
from astropy import units as u
from astropy import wcs as awcs
from karabo.simulation.utils import intersect2D


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
        self.shape: tuple = (0,0)
        self.sources: np.ndarray = None
        self.wcs: awcs = wcs
        if sources is not None:
            self.add_point_sources(sources)

    def add_point_sources(self, sources: np.ndarray):
        """
        Add new point sources to the sky model.

        :param sources: Array-like with shape (number of sources, 12). Each row representing one source.
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
        if 2 < sources.shape[1] < 14:
            if sources.shape[1] < 13:
                # if some elements are missing fill them up with zeros except `source_id`
                missing_shape = 13 - sources.shape[1]
                fill = np.hstack((np.zeros((sources.shape[0], 12)), np.array([[None]*sources.shape[0]]).reshape(-1,1)))
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
            return self[:,:-1]

    def filter_by_radius(self, inner_radius_deg: float, outer_radius_deg: float, ra0_deg: float, dec0_deg: float):
        """
        Filters the sky according the an inner and outer circle from the phase center

        :param inner_radius_deg: Inner radius in degrees
        :param outer_radius_deg: Outer raidus in degrees
        :param ra0_deg: Phase center right ascention
        :param dec0_deg: Phase center declination
        """
        inner_circle = SphericalCircle((ra0_deg*u.deg, dec0_deg*u.deg), inner_radius_deg*u.deg)
        outer_circle = SphericalCircle((ra0_deg*u.deg, dec0_deg*u.deg), outer_radius_deg*u.deg)
        outer_sources = outer_circle.contains_points(self[:,0:2]).astype('int')
        inner_sources = inner_circle.contains_points(self[:,0:2]).astype('int')
        filtered_sources = np.array(outer_sources - inner_sources, dtype='bool')
        filtered_sources_idxs = np.where(filtered_sources == True)[0]
        self.sources = self.sources[filtered_sources_idxs]
        self.__update_sky_model()

    def filter_by_flux(self, min_flux_jy: float, max_flux_jy: float):
        """
        Filters the sky using the Stokes-I-flux
        Values outside the range are removed

        :param min_flux_jy: Minimum flux in Jy
        :param max_flux_jy: Maximum flux in Jy
        """
        stokes_I_flux = self[:,2]
        filtered_sources_idxs = np.where(np.logical_and(stokes_I_flux <= max_flux_jy, stokes_I_flux >= min_flux_jy))[0]
        self.sources = self.sources[filtered_sources_idxs]
        self.__update_sky_model()

    def filter_by_frequency(self, min_freq: float, max_freq: float):
        """
        Filters the sky using the referency frequency in Hz

        :param min_freq: Minimum frequency in Hz
        :param max_freq: Maximum frequency in Hz
        """
        freq = self[:,6]
        filtered_sources_idxs = np.where(np.logical_and(freq <= max_freq, freq >= min_freq))[0]
        self.sources = self.sources[filtered_sources_idxs]
        self.__update_sky_model()

    def get_wcs(self) -> awcs:
        """
        Gets the currently active world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html
        """
        return self.wcs

    def set_wcs(self, wcs: awcs):
        """
        Sets a new world coordinate system astropy.wcs
        For details see https://docs.astropy.org/en/stable/wcs/index.html
        """
        self.wcs = wcs

    def __setup_default_wcs(self, phase_center: list = [0,0]):
        """
        Defines a default world coordinate system astropy.wcs
        For more details see https://docs.astropy.org/en/stable/wcs/index.html
        """
        w = awcs.wcs.WCS(naxis=2)
        w.wcs.crpix = [0, 0] # coordinate reference pixel per axis
        w.wcs.cdelt = [-1, 1] # coordinate increments on sphere per axis
        w.wcs.crval = phase_center
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"] # coordinate axis type
        self.wcs = w

    @staticmethod
    def get_fits_catalog(path: str) -> Table:
        """
        Gets astropy.table.table.Table from a .fits catalog

        :param path: Location of the .fits file

        :return: fits catalog
        """
        return Table.read(path)

    def explore_sky(self, phase_center: np.ndarray = np.array([0,0]), xlim: tuple = (-1,1), ylim: tuple = (-1,1), 
                 figsize: tuple = (6,6), title: str = '', xlabel: str = '', ylabel: str = '', wcs: awcs = None, 
                 s: float = 20, cfun: Callable = np.log10, cmap: str = 'plasma', cbar_label: str = '',
                 with_labels: bool = False):
        """
        A scatter plot of y vs. x of the point sources of the SkyModel

        :param phase_center:
        :param xlim:
        :param ylim:
        :param figsize:
        :param title:
        :param xlabel:
        :param ylabel:
        :param wcs:
        :param s:
        :param cfun:
        :param cmap:
        :param cbar_label:
        :param with_labels:
        """
        if wcs is None:
            if self.wcs is None:
                self.__setup_default_wcs(phase_center)
            wcs = self.wcs
        px, py = self.wcs.wcs_world2pix(self[:,0], self[:,1], 1) # ra-dec transformation

        flux, vmin, vmax = None, None, None
        if cmap is not None and cfun is not None:
            flux = self[:,2]
            flux = cfun(flux)
            vmin, vmax = np.min(flux), np.max(flux)
        else: # set both to None if one of them is None
            cfun = None
            cmap = None
        
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(px, py, s=s, c=flux, cmap=cmap, vmin=vmin, vmax=vmax)

        if with_labels:
            for i, txt in enumerate(self[:,-1]):
                ax.annotate(txt, (px[i], py[i]))

        plt.axis('equal')
        plt.colorbar(sc, label=cbar_label)
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def get_OSKAR_sky(self) -> oskar.Sky:
        """
        Get OSKAR sky model object from the defined Sky Model

        :return: oskar sky model
        """
        return oskar.Sky.from_array(self[:,:-1])

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
            pass # nothing toDo here
        return sources

    def __setitem__(self, key, value):
        """
        Allows to set values in an np.ndarray like manner

        :param key: slice key
        :param value: values to store
        """
        self.sources[key] = value