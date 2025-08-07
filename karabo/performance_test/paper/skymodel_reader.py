"""
Karabo offers sky surveys on a public CSCS repsoitory. But we cannot access
it from Euler. Thus, we need to transfer the files to Euler and read them in.
This module handles this for GLEAM and MIGHTEE sky surveys.
"""

import astropy.units as u

from karabo.simulation.sky_model import SkyModel, SkyPrefixMapping, SkySourcesUnits


def read_gleam_sky_from_fits(survey_file: str, min_freq=None, max_freq=None):
    encoded_freq = u.MHz
    unit_mapping = {
        "Jy": u.Jy,
        "Jy/beam": u.Jy / u.beam,
        "deg": u.deg,
        "arcsec": u.arcsec,
    }

    prefix_mapping = SkyPrefixMapping(
        ra="RAJ2000",
        dec="DEJ2000",
        stokes_i="Fp{0}",
        major="a{0}",
        minor="b{0}",
        pa="pa{0}",
        id="GLEAM",
    )

    units_sources = SkySourcesUnits(
        stokes_i=u.Jy / u.beam,
    )

    return SkyModel.get_sky_model_from_fits(
        fits_file=survey_file,
        prefix_mapping=prefix_mapping,
        unit_mapping=unit_mapping,
        units_sources=units_sources,
        min_freq=min_freq,
        max_freq=max_freq,
        encoded_freq=encoded_freq,
    )


def read_mightee_sky_from_fits(survey_file, min_freq=None, max_freq=None):
    unit_mapping = {
        "DEG": u.deg,
        "JY": u.Jy,
        "JY/BEAM": u.Jy / u.beam,
        "HZ": u.Hz,
    }
    prefix_mapping = SkyPrefixMapping(
        ra="RA",
        dec="DEC",
        stokes_i="S_PEAK",
        ref_freq="NU_EFF",
        major="IM_MAJ",
        minor="IM_MIN",
        pa="IM_PA",
        id="NAME",
    )
    units_sources = SkySourcesUnits(
        stokes_i=u.Jy / u.beam,
    )
    return SkyModel.get_sky_model_from_fits(
        fits_file=survey_file,
        prefix_mapping=prefix_mapping,
        unit_mapping=unit_mapping,
        units_sources=units_sources,
        min_freq=min_freq,
        max_freq=max_freq,
        encoded_freq=None,
        memmap=False,
    )
