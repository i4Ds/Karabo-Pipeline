import os
import unittest

import numpy as np
import xarray as xr

from karabo.data.external_data import (
    BATTYESurveyDownloadObject,
    ExampleHDF5Map,
    GLEAMSurveyDownloadObject,
    MIGHTEESurveyDownloadObject,
)
from karabo.simulation.sky_model import Polarisation, SkyModel
from karabo.test import data_path


class TestSkyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/"):
            os.makedirs("result/")

    def test_init(self):
        sky1 = SkyModel()
        sky_data = xr.DataArray(
            [
                [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0, "source1"],
                [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45, "source2"],
                [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10, "source3"],
            ]
        )
        sky1.add_point_sources(sky_data)
        sky2 = SkyModel(sky_data)
        # test if sources are inside now
        assert np.all(sky1.sources == sky2.sources)
        self.assertEqual(sky_data.shape, sky1.sources.shape)
        self.assertEqual(sky_data.shape, sky2.sources.shape)

    def test_not_full_array(self):
        sky1 = SkyModel()
        sky_data = xr.DataArray([[20.0, -30.0, 1], [20.0, -30.5, 3], [20.5, -30.5, 3]])
        sky1.add_point_sources(sky_data)
        sky2 = SkyModel(sky_data)
        # test if doc shape were expanded
        self.assertEqual(sky1.sources.shape, (sky_data.shape[0], 12))
        self.assertEqual(sky2.sources.shape, (sky_data.shape[0], 12))

    def test_plot_gleam(self):
        sky = SkyModel.get_GLEAM_Sky([76])
        sky.explore_sky([250, -80], s=0.1)
        cartesian_sky = sky.get_cartesian_sky()
        print(cartesian_sky)

    def test_filter_sky_model(self):
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
        assert len(filtered_sky.sources) == 8
        filtered_sky.write_to_file("./result/filtered_sky.csv")
        filtered_sky_euclidean_approx = (
            sky.filter_by_radius_euclidean_flat_approximation(
                0, 0.55, phase_center[0], phase_center[1]
            )
        )
        assert len(filtered_sky_euclidean_approx.sources) == len(filtered_sky.sources)

    def test_read_sky_model(self):
        sky = SkyModel.read_from_file(f"{data_path}/filtered_sky.csv")
        phase_center = [250, -80]  # ra,dec
        sky.explore_sky(
            phase_center=phase_center,
            figsize=(8, 6),
            s=80,
            xlim=(254, 246),  # RA-lim
            ylim=(-81, -79),  # DEC-lim
            with_labels=True,
        )

    def test_read_healpix_map(self):
        download = ExampleHDF5Map()
        path = download.get()
        source_array, _ = SkyModel.read_healpix_file_to_sky_model_array(
            f"{path}",
            0,
            Polarisation.STOKES_I,
        )
        sky = SkyModel(source_array)
        sky.explore_sky([250, -80])

    def test_get_poisson_sky(self):
        sky = SkyModel.get_random_poisson_disk_sky((220, -60), (260, -80), 0.1, 0.8, 2)
        sky.explore_sky([240, -70])

    def test_cscs_resource_availability(self):
        gleam = GLEAMSurveyDownloadObject()
        assert gleam.is_available()
        battye = BATTYESurveyDownloadObject()
        assert battye.is_available()
        mightee = MIGHTEESurveyDownloadObject()
        assert mightee.is_available()
        map = ExampleHDF5Map()
        assert map.is_available()

    def test_download_gleam_and_make_sky_model(self):
        sky = SkyModel.get_GLEAM_Sky([76])
        sky.explore_sky([250, -30], s=0.1)
        assert sky.num_sources > 0

    def test_transform_numpy_to_xarray(self):
        sources = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, "source1"],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, "source2"],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, "source3"],
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, "source4"],
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, "source5"],
            ],
            dtype=object,
        )
        sky = SkyModel(sources)
        assert isinstance(sky.sources, xr.DataArray)
        assert sky.num_sources > 0
        assert sky.to_np_array(with_obj_ids=True).shape == (sky.num_sources, 13)
