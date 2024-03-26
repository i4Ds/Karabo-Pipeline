import os
import tempfile
from copy import copy
from dataclasses import fields
from typing import Dict, TypedDict, get_args

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.io.fits import ColDefs, Column
from astropy.units import UnitBase, UnitConversionError
from numpy.typing import NDArray
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame

from karabo.data.external_data import (
    BATTYESurveyDownloadObject,
    DilutedBATTYESurveyDownloadObject,
    ExampleHDF5Map,
    GLEAMSurveyDownloadObject,
    HISourcesSmallCatalogDownloadObject,
    MGCLSContainerDownloadObject,
    MIGHTEESurveyDownloadObject,
)
from karabo.simulation.sky_model import (
    KaraboSkyModelError,
    Polarisation,
    SkyModel,
    SkyPrefixMapping,
    SkySourcesColName,
    SkySourcesUnits,
)
from karabo.simulator_backend import SimulatorBackend


@pytest.fixture(scope="function")
def gleam() -> SkyModel:
    return SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)


def test_download_gleam_and_make_sky_model():
    sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)
    sample_prefix_mapping = SkyPrefixMapping([], [], [])
    number_of_sky_attributes = len(sample_prefix_mapping.__dict__)

    assert sky.num_sources > 0

    # -1 since we do not return the source ID
    assert sky.to_np_array().shape == (sky.num_sources, number_of_sky_attributes - 1)
    assert sky.source_ids["dim_0"].shape[0] == sky.shape[0]  # checking source-ids


def test_mightee():
    _ = SkyModel.get_MIGHTEE_Sky()
    with pytest.raises(KaraboSkyModelError):
        _ = SkyModel.get_MIGHTEE_Sky(min_freq=3e11)
    _ = SkyModel.get_MIGHTEE_Sky(max_freq=3e11)
    _ = SkyModel.get_MIGHTEE_Sky(min_freq=1.3e9, max_freq=1.4e9)


def test_filter_sky_model(gleam: SkyModel):
    phase_center = [250, -80]  # ra,dec
    filtered_sky = gleam.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    assert len(filtered_sky.sources) == 8
    filtered_sky_euclidean_approx = gleam.filter_by_radius_euclidean_flat_approximation(
        0, 0.55, phase_center[0], phase_center[1]
    )
    assert len(filtered_sky_euclidean_approx.sources) == len(filtered_sky.sources)


def test_init(sky_data_with_ids: NDArray[np.object_]):
    sky1 = SkyModel()
    sky1.add_point_sources(sky_data_with_ids)
    sky2 = SkyModel(sky_data_with_ids)
    # test if sources are inside now (-1 because ids are in `xarray.DataArray.coord`)
    assert sky_data_with_ids.shape[1] - 1 == sky1.sources.shape[1]
    assert sky_data_with_ids.shape[1] - 1 == sky2.sources.shape[1]


def test_not_full_array():
    sky1 = SkyModel()
    sky_data = xr.DataArray([[20.0, -30.0, 1], [20.0, -30.5, 3], [20.5, -30.5, 3]])
    sky1.add_point_sources(sky_data)
    sky2 = SkyModel(sky_data)
    # test if doc shape were expanded
    assert sky1.sources.shape == (sky_data.shape[0], 14)
    assert sky2.sources.shape == (sky_data.shape[0], 14)


def test_filter_sky_model_h5():
    sky = SkyModel.get_sample_simulated_catalog()
    phase_center = [21.44213503, -30.70729488]
    filtered_sky = sky.filter_by_radius_euclidean_flat_approximation(
        0, 1, phase_center[0], phase_center[1]
    )
    filtered_sky.setup_default_wcs(phase_center)
    assert len(filtered_sky.sources) == 33
    assert np.all(
        np.abs(filtered_sky.sources.compute()[:, 0:2] - phase_center) < [2, 2]
    )


def test_filter_flux_sky_model(sky_data_with_ids: NDArray[np.object_]):
    flux_min = 1
    flux_max = 2
    sky = SkyModel(sky_data_with_ids)
    assert (
        sky.sources[:, 2].min() < flux_min or sky.sources[:, 2].max() > flux_max
    ), "Test data not correct"
    assert len(sky.sources) > 0, "Test data not correct"
    filtered_sky = sky.filter_by_flux(flux_min, flux_max)
    assert np.all((filtered_sky[:, 2] >= flux_min) & (filtered_sky[:, 2] <= flux_max))


def test_read_sky_model(gleam: SkyModel):
    with tempfile.TemporaryDirectory() as tmpdir:
        sky_path = os.path.join(tmpdir, "gleam.csv")
        gleam.save_sky_model_as_csv(path=sky_path)
        sky2 = SkyModel.read_from_file(path=sky_path)
        assert gleam.sources.shape == sky2.sources.shape


def test_get_cartesian(sky_data_with_ids: NDArray[np.object_]):
    sky1 = SkyModel()
    sky1.add_point_sources(sky_data_with_ids)
    cart_sky = sky1.get_cartesian_sky()
    print(cart_sky)


def test_cscs_resource_availability():
    gleam = GLEAMSurveyDownloadObject()
    assert gleam.is_available()
    with pytest.raises(NotImplementedError):
        BATTYESurveyDownloadObject()
    with pytest.raises(NotImplementedError):
        DilutedBATTYESurveyDownloadObject()
    sample_sky = HISourcesSmallCatalogDownloadObject()
    assert sample_sky.is_available()
    mightee = MIGHTEESurveyDownloadObject()
    assert mightee.is_available()
    map = ExampleHDF5Map()
    assert map.is_available()
    mgcls = MGCLSContainerDownloadObject(".+")
    assert mgcls.is_available()


def test_read_write_sky_model(sky_data: NDArray[np.float64]):
    sky = SkyModel(sky_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        sky_path = os.path.join(tmpdir, "sky.csv")
        sky.write_to_file(sky_path)
        sky_loaded = sky.read_from_file(sky_path)
        assert np.all(sky.sources == sky_loaded.sources)


def test_read_healpix_map():
    download = ExampleHDF5Map()
    path = download.get()
    source_array, _ = SkyModel.read_healpix_file_to_sky_model_array(
        f"{path}",
        0,
        Polarisation.STOKES_I,
    )
    _ = SkyModel(source_array)


def test_get_poisson_sky():
    _ = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 0.1, 0.8, 2)


def test_explore_sky(gleam: SkyModel):
    gleam.explore_sky([250, -80], s=0.1)


def test_convert_sky_to_backends():
    # Create test sky with all sources at redshift 1,
    # which corresponds to 21cm frequency of ~713 MHz.
    # Then, request RASCIL list of SkyComponents for these sources,
    # and verify the following:
    # 1. RA and Dec positions are maintained
    # 2. Frequency and fluxes are correctly assigned based on redshift
    # 3. Other info (polarisation, shape) are correct
    FLUX = 20

    sky = SkyModel.sky_test()
    sky.sources[:, 2] = FLUX  # Manually override fluxes
    sky.sources[:, 13] = 1  # Manually set all redshifts to 1

    # Sources have redshift 1, i.e. frequency ~ 713 MHz, and thus
    # all sources should fall on bin index 1 in the array below
    # (i.e. within the channel starting at 710 MHz)
    desired_frequencies_hz = 1e6 * np.array([700, 710, 720, 730, 740, 750])
    expected_channel_index = 1

    # Verify that converting to OSKAR backend is a no-op
    oskar_sky = sky.convert_to_backend(backend=SimulatorBackend.OSKAR)
    assert np.allclose(oskar_sky.sources, sky.sources)

    # Verify conversion to RASCIL backend
    rascil_sky = sky.convert_to_backend(
        backend=SimulatorBackend.RASCIL,
        desired_frequencies_hz=desired_frequencies_hz,
    )

    assert len(rascil_sky) == sky.sources.shape[0]
    for i, rascil_component in enumerate(rascil_sky):
        # Verify that RA, Dec and other parameters are correct
        assert np.isclose(rascil_component.direction.ra.value, sky.sources[i][0])
        assert np.isclose(rascil_component.direction.dec.value, sky.sources[i][1])
        assert rascil_component.shape == "Point"
        assert rascil_component.polarisation_frame == PolarisationFrame("stokesI")

        # Verify flux and frequencies of the source
        # Assert source frequencies are the same as desired frequency channel starts,
        # excluding the last entry of the desired frequencies
        # (since it marks the end of the last frequency channel)
        assert len(rascil_component.frequency) == len(desired_frequencies_hz) - 1
        assert np.allclose(rascil_component.frequency, desired_frequencies_hz[:-1])

        # Assert only one flux entry is non-zero
        assert (
            sum(~np.isclose(0, rascil_component.flux)) == 1
        )  # Only one entry is non-zero

        # Assert that non-zero flux entry is in the expected frequency channel
        assert np.isclose(FLUX, rascil_component.flux[expected_channel_index])


def test_SkySourcesColName_assumption():
    sky_sources_col_names = get_args(SkySourcesColName)
    for field in fields(SkyPrefixMapping):
        assert field.name in sky_sources_col_names
    for field in fields(SkySourcesUnits):
        assert field.name in sky_sources_col_names


class _ColsMappingDict(TypedDict):
    cols: ColDefs
    unit_mapping: Dict[str, UnitBase]
    prefix_mapping: SkyPrefixMapping
    sky_sources_units: SkySourcesUnits
    n_encoded_freqs: int
    n_formattable_cols: int
    n_unit_cols: int


@pytest.fixture(scope="function")
def formattable() -> _ColsMappingDict:
    """Matching `_ColsMappingDict` of formattable col-names."""
    cols = ColDefs(
        [
            Column(name="NAME", format="14A", disp="A14"),
            Column(name="RAJ2000", format="D", unit="deg", disp="F10.6"),
            Column(name="DEJ2000", format="D", unit="deg", disp="F10.6"),
            Column(name="Fp076", format="D", unit="Jy/beam", disp="F11.6"),
            Column(name="a076", format="D", unit="arcsec", disp="D12.5"),
            Column(name="b076", format="E", unit="arcsec", disp="F7.3"),
            Column(name="pa076", format="D", unit="deg", disp="F10.6"),
            Column(name="Fp084", format="D", unit="Jy/beam", disp="F11.6"),
            Column(name="a084", format="D", unit="arcsec", disp="D12.5"),
            Column(name="b084", format="E", unit="arcsec", disp="F7.3"),
            Column(name="pa084", format="D", unit="deg", disp="F10.6"),
        ]
    )
    unit_mapping = {
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
        id="NAME",
    )
    units_sources = SkySourcesUnits(
        stokes_i=u.Jy / u.beam,
    )
    cols_mapping: _ColsMappingDict = {
        "cols": cols,
        "unit_mapping": unit_mapping,
        "prefix_mapping": prefix_mapping,
        "sky_sources_units": units_sources,
        "n_encoded_freqs": 2,
        "n_formattable_cols": 4,
        "n_unit_cols": 10,
    }
    return cols_mapping


@pytest.fixture(scope="function")
def non_formattable() -> _ColsMappingDict:
    """Matching `_ColsMappingType` of non-formattable col-names."""
    cols = ColDefs(
        [
            Column(name="RA", format="E", unit="DEG"),
            Column(name="DEC", format="E", unit="DEG"),
            Column(name="S_PEAK", format="E", unit="JY/BEAM"),
            Column(name="NU_EFF", format="E", unit="HZ"),
            Column(name="IM_MAJ", format="E", unit="DEG"),
            Column(name="IM_MIN", format="E", unit="DEG"),
            Column(name="IM_PA", format="E", unit="DEG"),
            Column(name="NAME", format="19A"),
        ]
    )
    unit_mapping: Dict[str, UnitBase] = {
        "DEG": u.deg,
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
    cols_mapping: _ColsMappingDict = {
        "cols": cols,
        "unit_mapping": unit_mapping,
        "prefix_mapping": prefix_mapping,
        "sky_sources_units": units_sources,
        "n_encoded_freqs": 0,
        "n_formattable_cols": 0,
        "n_unit_cols": 7,
    }
    return cols_mapping


def test_format_prefix_freq_mapping(
    formattable: _ColsMappingDict,
    non_formattable: _ColsMappingDict,
):
    (
        prefix_mapping,
        num_formattings,
        _,
    ) = SkySourcesUnits.format_sky_prefix_freq_mapping(
        cols=non_formattable["cols"],
        prefix_mapping=non_formattable["prefix_mapping"],
        encoded_freq=None,
    )
    assert prefix_mapping is non_formattable["prefix_mapping"]
    assert num_formattings == 0
    with pytest.raises(RuntimeError):
        _ = SkySourcesUnits.format_sky_prefix_freq_mapping(
            cols=non_formattable["cols"],
            prefix_mapping=non_formattable["prefix_mapping"],
            encoded_freq=u.MHz,
        )
    (
        prefix_mapping,
        num_formattings,
        names_and_freqs,
    ) = SkySourcesUnits.format_sky_prefix_freq_mapping(
        cols=formattable["cols"],
        prefix_mapping=formattable["prefix_mapping"],
        encoded_freq=u.MHz,
    )
    assert prefix_mapping != formattable["prefix_mapping"]
    assert num_formattings == formattable["n_encoded_freqs"]
    assert (
        len(names_and_freqs)
        == formattable["n_encoded_freqs"] * formattable["n_formattable_cols"]
    )


def test_get_unit_scales(
    formattable: _ColsMappingDict,
    non_formattable: _ColsMappingDict,
):
    with pytest.raises(RuntimeError):  # because formattable-strings are not col-names
        _ = formattable["sky_sources_units"].get_unit_scales(
            cols=formattable["cols"],
            unit_mapping=formattable["unit_mapping"],
            prefix_mapping=formattable["prefix_mapping"],
        )
    unit_scales_non_formattable = non_formattable["sky_sources_units"].get_unit_scales(
        cols=non_formattable["cols"],
        unit_mapping=non_formattable["unit_mapping"],
        prefix_mapping=non_formattable["prefix_mapping"],
    )
    assert len(unit_scales_non_formattable) == non_formattable["n_unit_cols"]
    unit_mapping_typo = copy(non_formattable["unit_mapping"])
    unit_mapping_typo.update({"non-existing-unit-str": u.km})  # u.km doesn't rly matter
    with pytest.raises(RuntimeError):
        _ = non_formattable["sky_sources_units"].get_unit_scales(
            cols=non_formattable["cols"],
            unit_mapping=unit_mapping_typo,
            prefix_mapping=non_formattable["prefix_mapping"],
        )
    prefix_mapping_typo = copy(non_formattable["prefix_mapping"])
    prefix_mapping_typo.ra = "definitely-not-ra"
    with pytest.raises(RuntimeError):
        _ = non_formattable["sky_sources_units"].get_unit_scales(
            cols=non_formattable["cols"],
            unit_mapping=non_formattable["unit_mapping"],
            prefix_mapping=prefix_mapping_typo,
        )


def test_extract_names_and_freqs(
    formattable: _ColsMappingDict,
):
    naf_formattable = SkySourcesUnits.extract_names_and_freqs(
        string=formattable["prefix_mapping"].stokes_i,
        cols=formattable["cols"],
        unit=u.MHz,
    )
    assert len(naf_formattable) == formattable["n_encoded_freqs"]
    naf_formattable2 = SkySourcesUnits.extract_names_and_freqs(
        string=formattable["prefix_mapping"].stokes_i,
        cols=formattable["cols"],
        unit=u.Hz,
    )
    assert all(
        np.array(list(naf_formattable.values()))
        / np.array(list(naf_formattable2.values()))
        == (u.MHz.to(u.Hz))  # 1e6
    )
    with pytest.raises(UnitConversionError):
        _ = SkySourcesUnits.extract_names_and_freqs(
            string=formattable["prefix_mapping"].stokes_i,
            cols=formattable["cols"],
            unit=u.km,
        )
    with pytest.raises(ValueError):
        _ = SkySourcesUnits.extract_names_and_freqs(
            string=formattable["prefix_mapping"].ra,
            cols=formattable["cols"],
            unit=u.GHz,
        )
