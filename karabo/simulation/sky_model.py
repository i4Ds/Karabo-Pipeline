import numpy as np
import oskar


class SkyModel:
    def __init__(self):
        self.sources = None

    def add_points_sources(self, sources: np.array):
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
        new_sources = np.array(
            [[right_ascension, declination, stokes_I_flux, stokes_Q_flux, stokes_U_flux,
              stokes_V_flux, reference_frequency, spectral_index, rotation_measure,
              major_axis_FWHM, minor_axis_FWHM, position_angle]])
        if self.sources is not None:
            self.sources = np.vstack(self.sources, new_sources)
        else:
            self.sources = new_sources

    def get_OSKAR_sky(self) -> oskar.Sky:
        return oskar.Sky.from_array(self.sources)
