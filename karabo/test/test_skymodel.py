import os
import tempfile

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

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


def test_explore_sky():
    sky = SkyModel.get_GLEAM_Sky([76])
    sky.explore_sky([250, -80], s=0.1)
