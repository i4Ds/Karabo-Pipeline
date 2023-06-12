import os
import tempfile

import numpy as np
from numpy.typing import NDArray

from karabo.data.external_data import ExampleHDF5Map
from karabo.simulation.sky_model import Polarisation, SkyModel


def test_init(sky_data_with_ids: NDArray[np.object_]):
    sky1 = SkyModel()
    sky1.add_point_sources(sky_data_with_ids)
    sky2 = SkyModel(sky_data_with_ids)
    # test if sources are inside now
    assert sky_data_with_ids.shape == sky1.sources.shape
    assert sky_data_with_ids.shape == sky2.sources.shape


def test_not_full_array():
    sky1 = SkyModel()
    sky_data = np.array([[20.0, -30.0, 1], [20.0, -30.5, 3], [20.5, -30.5, 3]])
    sky1.add_point_sources(sky_data)
    sky2 = SkyModel(sky_data)
    # test if doc shape were expanded
    assert sky1.sources.shape == (sky_data.shape[0], 13)
    assert sky2.sources.shape == (sky_data.shape[0], 13)


def test_plot_gleam():
    sky = SkyModel.get_GLEAM_Sky([76])
    sky.explore_sky([250, -80], s=0.1)
    cartesian_sky = sky.get_cartesian_sky()
    print(cartesian_sky)


def test_get_cartesian(sky_data_with_ids: NDArray[np.object_]):
    sky1 = SkyModel()
    sky1.add_point_sources(sky_data_with_ids)
    cart_sky = sky1.get_cartesian_sky()
    print(cart_sky)


def test_filter_sky_model():
    sky = SkyModel.get_GLEAM_Sky([76])
    phase_center = [250, -80]  # ra,dec
    filtered_sky = sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    filtered_sky.setup_default_wcs(phase_center)
    filtered_sky.explore_sky(
        phase_center=phase_center,
        figsize=(8, 6),
        s=80,
        xlim=(254, 246),  # RA-lim
        ylim=(-81, -79),  # DEC-lim
        with_labels=True,
    )


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
    source_array, nside = SkyModel.read_healpix_file_to_sky_model_array(
        f"{path}",
        0,
        Polarisation.STOKES_I,
    )
    sky = SkyModel(source_array)
    sky.explore_sky([250, -80])


def test_get_poisson_sky():
    sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 0.1, 0.8, 2)
    sky.explore_sky([240, -70])
