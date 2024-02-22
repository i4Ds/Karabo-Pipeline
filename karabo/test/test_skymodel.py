import os
import tempfile

import numpy as np
import pytest
import xarray as xr
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
from karabo.simulation.sky_model import Polarisation, SkyModel
from karabo.simulator_backend import SimulatorBackend


def test_filter_sky_model():
    sky = SkyModel.get_GLEAM_Sky([76])
    phase_center = [250, -80]  # ra,dec
    filtered_sky = sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    filtered_sky.explore_sky(
        phase_center=phase_center,
        figsize=(8, 6),
        s=80,
        xlim=(254, 246),  # RA-lim
        ylim=(-81, -79),  # DEC-lim
        with_labels=True,
    )
    assert len(filtered_sky.sources) == 8
    filtered_sky_euclidean_approx = sky.filter_by_radius_euclidean_flat_approximation(
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
    filtered_sky.explore_sky(
        phase_center,
        s=1,
        cmap="jet",
        cbar_label="Flux [Jy]",
        cfun=None,
        wcs_enabled=False,
        xlabel="RA [deg]",
        ylabel="DEC [deg]",
    )
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


def test_read_sky_model():
    sky = SkyModel.get_GLEAM_Sky([76])
    with tempfile.TemporaryDirectory() as tmpdir:
        sky_path = os.path.join(tmpdir, "gleam.csv")
        sky.save_sky_model_as_csv(path=sky_path)
        sky2 = SkyModel.read_from_file(path=sky_path)
        assert sky.sources.shape == sky2.sources.shape


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
