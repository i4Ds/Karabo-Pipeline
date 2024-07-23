from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from rfc3986.exceptions import InvalidComponentsError

from karabo.data.external_data import (
    SingleFileDownloadObject,
    cscs_karabo_public_testing_base_url,
)
from karabo.data.obscore import FitsHeaderAxes, FitsHeaderAxis, ObsCoreMeta
from karabo.imaging.image import Image
from karabo.simulation.observation import Observation
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend


@pytest.fixture(scope="module")
def minimal_visibility() -> Visibility:
    vis_path = SingleFileDownloadObject(
        remote_file_path="test_minimal_visibility.vis",
        remote_base_url=cscs_karabo_public_testing_base_url,
    ).get()
    return Visibility(vis_path=vis_path)


@pytest.fixture(scope="module")
def minimal_fits_restored() -> Image:
    restored_path = SingleFileDownloadObject(
        remote_file_path="test_minimal_clean_restored.fits",
        remote_base_url=cscs_karabo_public_testing_base_url,
    ).get()
    return Image(path=restored_path)


class TestObsCoreMeta:
    def test_sshapes(self) -> None:
        assert (
            ObsCoreMeta.scircle(point=(214.2344, -21.5), radius=1.46)
            == "<(214.234d,-21.5d),1.46d>"
        )
        assert (
            ObsCoreMeta.spoint(point=(214.2344, -21.5), ndigits=4)
            == "(214.2344d,-21.5d)"
        )
        assert (
            ObsCoreMeta.spoly(poly=((0.0, 0.0), (100.0, -21.4), (332.1, 87.34)))
            == "{(0.0d,0.0d),(100.0d,-21.4d),(332.1d,87.34d)}"
        )

        with pytest.raises(ValueError):
            ObsCoreMeta.spoint(point=(214.2344, -90.1), ndigits=4)
            ObsCoreMeta.spoint(point=(214.2344, 90.1), ndigits=4)
            ObsCoreMeta.scircle(point=(214.2344, -21.5), radius=-1.0)
            ObsCoreMeta.scircle(point=(214.2344, -21.5), radius=181.4)
            ObsCoreMeta.spoly(poly=((0.0, 0.0), (100.0, -21.4), (332.1, 97.34)))
        with pytest.raises(RuntimeError):
            ObsCoreMeta.spoly(poly=((0.0, 0.0), (100.0, -21.4)))

    @pytest.mark.parametrize(
        ("authority", "path", "query", "fragment", "expected"),
        [
            ("", None, None, None, pytest.raises(ValueError)),  # authority < 3
            ("sk", None, None, None, pytest.raises(ValueError)),  # authority < 3
            ("ska", None, None, None, "ivo://ska"),
            ("éka", None, None, None, pytest.raises(ValueError)),  # start is not alphan
            ("skao", None, None, None, "ivo://skao"),
            ("skao", "~", None, None, pytest.raises(ValueError)),  # path must start w /
            ("skao", "/~", None, None, "ivo://skao/~"),
            ("skao", "/~%", None, None, pytest.raises(InvalidComponentsError)),
            ("skao", "/~/sth", None, None, "ivo://skao/~/sth"),
            ("skao", "/~/sth@", None, None, pytest.raises(ValueError)),  # has @ in path
            ("skao", "/~/sth:", None, None, pytest.raises(ValueError)),  # has : in path
            ("skao", None, "karabo", None, "ivo://skao?karabo"),
            ("skao", "/~", "karabo", None, "ivo://skao/~?karabo"),
            ("skao", "/~", "karabo:image.fits", None, "ivo://skao/~?karabo:image.fits"),
            (
                "skao",
                "/~",
                "karabo:image.fits",
                "header",
                "ivo://skao/~?karabo:image.fits#header",
            ),
        ],
    )
    def test_ivoid(
        self,
        authority: str,
        path: str,
        query: str,
        fragment: str,
        expected: Any,
    ) -> None:
        if isinstance(expected, str):
            assert (
                ObsCoreMeta.get_ivoid(
                    authority=authority,
                    path=path,
                    query=query,
                    fragment=fragment,
                )
                == expected
            )
        else:
            with expected:
                _ = ObsCoreMeta.get_ivoid(
                    authority=authority,
                    path=path,
                    query=query,
                    fragment=fragment,
                )

    def test_from_visibility(self, minimal_visibility: Visibility) -> None:
        telescope = Telescope.constructor("ASKAP", backend=SimulatorBackend.OSKAR)
        observation = Observation(  # settings from notebook, of `minimal_visibility`
            start_frequency_hz=100e6,
            start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
            phase_centre_ra_deg=250.0,
            phase_centre_dec_deg=-80.0,
            number_of_channels=16,
            frequency_increment_hz=1e6,
            number_of_time_steps=24,
        )
        ocm = ObsCoreMeta.from_visibility(
            vis=minimal_visibility,
            calibrated=False,
            tel=telescope,
            obs=observation,
        )
        assert ocm.dataproduct_type == "visibility"
        assert ocm.s_ra is not None and np.allclose(ocm.s_ra, 250.0)
        assert ocm.s_dec is not None and np.allclose(ocm.s_dec, -80.0)
        assert ocm.t_min is not None
        assert ocm.t_max is not None and ocm.t_max > ocm.t_min
        assert ocm.t_exptime is not None and ocm.t_exptime > 0.0
        assert ocm.t_resolution is not None and ocm.t_resolution > 0.0
        assert ocm.t_xel is not None
        assert ocm.em_min is not None and ocm.em_min > 0.0
        assert (
            ocm.em_max is not None and ocm.em_max > 0.0 and ocm.em_max <= ocm.em_min
        )  # <= because max-freq = min-wavelength
        assert ocm.em_xel is not None and ocm.em_xel >= 1
        assert ocm.access_estsize is not None and ocm.access_estsize > 0.0
        assert ocm.em_ucd is not None
        assert ocm.o_ucd is not None
        assert ocm.calib_level == 1  # because `calibrated` flag set to False
        assert ocm.instrument_name == telescope.name
        assert ocm.s_resolution is not None and ocm.s_resolution > 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "obscore-vis.json")
            with pytest.warns(UserWarning):  # mandatory fields not set
                json_str = ocm.to_json()
                assert len(json_str) > 1
            ocm.obs_collection = "<obs-collection>"
            ocm.obs_id = "<obs-id>"
            ocm.obs_publisher_did = "<obs-publisher-did>"
            _ = ocm.to_json(fpath=meta_path)
            assert os.path.exists(meta_path)

    def test_from_image(self, minimal_fits_restored: Image) -> None:
        axes = FitsHeaderAxes(freq=FitsHeaderAxis(axis=4, unit=u.Hz))
        ocm = ObsCoreMeta.from_image(img=minimal_fits_restored, fits_axes=axes)
        assert ocm.calib_level == 3
        assert ocm.dataproduct_type is not None and ocm.dataproduct_type == "image"
        assert ocm.access_format is not None and ocm.access_format == "image/fits"
        assert ocm.s_xel1 is not None and ocm.s_xel1 >= 1
        assert ocm.s_xel2 is not None and ocm.s_xel2 >= 1
        assert ocm.s_pixel_scale is not None and ocm.s_pixel_scale > 0.0
        assert ocm.s_fov is not None and ocm.s_fov > 0.0
        assert ocm.s_region is not None
        assert ocm.em_min is not None and ocm.em_min > 0.0
        assert (
            ocm.em_max is not None and ocm.em_max > 0.0 and ocm.em_max <= ocm.em_min
        )  # <= because max-freq = min-wavelength
        assert ocm.em_xel is not None and ocm.em_xel >= 1
        assert ocm.em_ucd is not None and ocm.em_ucd == "em.freq;em.radio"
        assert ocm.access_estsize is not None and ocm.access_estsize > 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "obscore-img.json")
            with pytest.warns(UserWarning):  # mandatory fields not set
                json_str = ocm.to_json()
                assert len(json_str) > 1
            ocm.obs_collection = "<obs-collection>"
            ocm.obs_id = "<obs-id>"
            ocm.obs_publisher_did = "<obs-publisher-did>"
            _ = ocm.to_json(fpath=meta_path)
            assert os.path.exists(meta_path)
