from os import stat
import numpy as np
import pandas as pd
import oskar
from astropy.table import Table


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

    """
    def __init__(self):
        self.sources = None
        self.num_sources = 0

    def add_points_sources(self, sources: np.ndarray):
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

        """
        if len(sources.shape) > 2:
            return
        if 2 < sources.shape[1] < 13:
            if sources.shape[1] < 12:
                # if some elements are missing fill them up with zeros
                missing_shape = 12 - sources.shape[1]
                fill = np.zeros((sources.shape[0], 12))
                fill[:, :-missing_shape] = sources
                sources = fill
            if self.sources is not None:
                self.sources = np.vstack((self.sources, sources))
            else:
                self.sources = sources
            self.num_sources += sources.shape[0]

    def add_point_source(self, right_ascension: float, declination: float, stokes_I_flux: float = 0,
                         stokes_Q_flux: float = 0,
                         stokes_U_flux: float = 0, stokes_V_flux: float = 0, reference_frequency: float = 0,
                         spectral_index: float = 0, rotation_measure: float = 0, major_axis_FWHM: float = 0,
                         minor_axis_FWHM: float = 0,
                         position_angle: float = 0):
        """
        Add a new point source to the sky model.

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
        """
        new_sources = np.array(
            [[right_ascension, declination, stokes_I_flux, stokes_Q_flux, stokes_U_flux,
              stokes_V_flux, reference_frequency, spectral_index, rotation_measure,
              major_axis_FWHM, minor_axis_FWHM, position_angle]])
        if self.sources is not None:
            self.sources = np.vstack(self.sources, new_sources)
        else:
            self.sources = new_sources
        self.num_sources += 1

    def to_array(self) -> np.ndarray:
        """
        Gets the sources as np.ndarray

        :return: self.sources
        """
        return self.sources

    def filter_by_radius(self, inner_radius_deg: float, outer_radius_deg: float, ra0_deg: float, dec0_deg: float):
        """
        Filters the sky according the an inner and outer circle from the phase center

        :param inner_radius_deg: Inner radius in degrees
        :param outer_radius_deg: Outer raidus in degrees
        :param ra0_deg: Phase center right ascention
        :param dec0_deg: Phase center declination
        """
        OSKAR_sky = self.get_OSKAR_sky()
        OSKAR_sky.filter_by_radius(inner_radius_deg, outer_radius_deg, ra0_deg, dec0_deg)
        self.sources = OSKAR_sky.to_array()
        self.num_sources = self.sources.shape[0]

    def filter_by_flux(self, min_flux_jy: float, max_flux_jy: float):
        """
        Filters the sky using the Stokes-I-flux
        Values outside the range are removed

        :param min_flux_jy: Minimum flux in Jy
        :param max_flux_jy: Maximum flux in Jy
        """
        stokes_I_flux = self.sources[:,2]
        idxs = np.where(np.logical_and(stokes_I_flux <= max_flux_jy, stokes_I_flux >= min_flux_jy))[0]
        self.sources = self.sources[idxs]
        self.num_sources = self.sources.shape[0]

    @staticmethod
    def get_fits_catalog(path: str) -> Table:
        """
        Gets astropy.table.table.Table from a .fits catalog

        :param path: Location of the .fits file

        :return: fits catalog
        """
        return Table.read(path)

    def get_OSKAR_sky(self) -> oskar.Sky:
        """
        Get OSKAR sky model object from the defined Sky Model

        :return: oskar sky model
        """
        # what about precision = "single"?
        return oskar.Sky.from_array(self.sources)
