import numpy as np
import oskar


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

    def add_points_sources(self, sources: np.array):
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

    def get_OSKAR_sky(self) -> oskar.Sky:
        """
        Get OSKAR sky model object from the defined Sky Model

        :return: oskar sky model
        """
        return oskar.Sky.from_array(self.sources)
