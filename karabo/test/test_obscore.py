from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time
from astropy import constants as const
from pytest import FixtureRequest
from rfc3986.exceptions import InvalidComponentsError

from karabo.data.obscore import FitsHeaderAxes, FitsHeaderAxis, ObsCoreMeta
from karabo.data.src import RucioMeta
from karabo.imaging.image import Image
from karabo.simulation.observation import Observation
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.simulator_backend import SimulatorBackend
from karabo.util.helpers import get_rnd_str


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

    @pytest.mark.parametrize(
        "vis_fixture_name",
        [
            "minimal_oskar_vis",
            "minimal_casa_ms",
        ],
    )
    def test_from_visibility(
        self,
        vis_fixture_name: str,
        request: FixtureRequest,
    ) -> None:
        visibility: Visibility = request.getfixturevalue(vis_fixture_name)
        exp_time_start, exp_ntimes = datetime(2024, 3, 15, 10, 46, 0), 24
        exp_time_res = 600.0
        exp_freq_start, exp_freq_res, exp_nfreqs = 100e6, 1e6, 16
        exp_em_max = const.c.value / 99.5e6
        exp_em_min = const.c.value / 115.5e6
        exp_ra, exp_dec = 250.0, -80.0
        if visibility.format == "OSKAR_VIS":
            telescope = Telescope.constructor("ASKAP", backend=SimulatorBackend.OSKAR)
            observation = Observation(  # original settings for `minimal_oskar_vis`
                start_frequency_hz=exp_freq_start,
                start_date_and_time=exp_time_start,
                phase_centre_ra_deg=exp_ra,
                phase_centre_dec_deg=exp_dec,
                number_of_channels=exp_nfreqs,
                frequency_increment_hz=exp_freq_res,
                number_of_time_steps=exp_ntimes,
            )
        else:
            telescope = None
            observation = None
        ocm = ObsCoreMeta.from_visibility(
            vis=visibility,
            calibrated=False,
            tel=telescope,
            obs=observation,
        )
        assert ocm.dataproduct_type == "visibility"
        assert ocm.s_ra is not None and np.allclose(ocm.s_ra, exp_ra)
        assert ocm.s_dec is not None and np.allclose(ocm.s_dec, exp_dec)
        assert ocm.t_min is not None and np.isclose(ocm.t_min, Time(exp_time_start).mjd)
        assert ocm.t_max is not None and ocm.t_max > ocm.t_min
        assert ocm.t_exptime is not None and ocm.t_exptime > 0.0
        assert ocm.t_resolution is not None and ocm.t_resolution > 0.0
        assert ocm.t_exptime >= ocm.t_resolution
        assert np.isclose(ocm.t_resolution, exp_time_res)
        assert np.isclose(ocm.t_exptime, exp_ntimes * exp_time_res)
        assert ocm.t_xel is not None and ocm.t_xel == exp_ntimes
        assert ocm.em_min is not None and ocm.em_min > 0.0
        assert ocm.em_max is not None and ocm.em_max > 0.0
        assert ocm.em_max > ocm.em_min
        if visibility.format == "OSKAR_VIS":
            # there's an issue with minimal_casa_ms
            # > tb.open('test_minimal_casa.ms/SPECTRAL_WINDOW')
            # > tb.getcol('CHAN_FREQ')
            # array([[7.600e+07],[1.760e+08],...[1.476e+09],[1.576e+09]])
            assert ocm.em_min == exp_em_min
            assert ocm.em_max == exp_em_max
        assert ocm.em_xel is not None and ocm.em_xel == exp_nfreqs
        assert ocm.access_estsize is not None and ocm.access_estsize > 0.0
        assert ocm.em_ucd is not None
        assert ocm.o_ucd is not None
        assert ocm.calib_level == 1  # because `calibrated` flag set to False
        assert ocm.instrument_name is not None
        assert ocm.s_resolution is not None and ocm.s_resolution > 0.0
        assert ocm.em_res_power is not None and ocm.em_res_power > 1.0 # λ/δλ>1

        allowed_pols = [
            "I", "Q", "U", "V", "RR", "LL", "RL", "LR", "XX", "YY", "XY", "YX",
            "POLI", "POLA"
        ]
        if visibility.format != "OSKAR_VIS":
            assert ocm.pol_xel is not None and ocm.pol_xel > 0
            assert ocm.pol_states is not None and len(ocm.pol_states) > 0
            assert ocm.pol_states.startswith("/")
            assert ocm.pol_states.endswith("/")
            pols = [*filter(None, ocm.pol_states.split("/"))]
            assert len(pols) == ocm.pol_xel
            for pol in pols:
                assert pol in allowed_pols

        # test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "obscore-vis.json")
            with pytest.warns(UserWarning):  # mandatory fields not set
                _ = ocm.to_dict()
            ocm.obs_collection = "<obs-collection>"
            ocm.obs_id = "<obs-id>"
            ocm.obs_publisher_did = "<obs-publisher-did>"
            _ = ocm.to_dict(fpath=meta_path)
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
                _ = ocm.to_dict()
            ocm.obs_collection = "<obs-collection>"
            ocm.obs_id = "<obs-id>"
            ocm.obs_publisher_did = "<obs-publisher-did>"
            _ = ocm.to_dict(fpath=meta_path)
            assert os.path.exists(meta_path)


class TestRucioMeta:
    def test_json_creation(self, minimal_fits_restored: Image) -> None:
        axes = FitsHeaderAxes(freq=FitsHeaderAxis(axis=4, unit=u.Hz))
        ocm = ObsCoreMeta.from_image(img=minimal_fits_restored, fits_axes=axes)
        name = os.path.split(minimal_fits_restored.path)[-1]
        rm = RucioMeta(
            namespace="karabo-sim",
            name=name,
            lifetime=86400,  # 1 day
            dataset_name=None,
            meta=ocm,
        )
        ocm.obs_collection = "MRO/ASKAP"
        obs_sim_id = 0  # unique observation-simulation ID of `USER`
        user_rnd_str = get_rnd_str(k=10, seed=os.environ.get("USER"))
        ocm.obs_id = f"karabo-{user_rnd_str}-{obs_sim_id}"
        obs_publisher_did = RucioMeta.get_ivoid(  # rest args are defaults
            namespace=rm.namespace,
            name=rm.name,
        )
        ocm.obs_publisher_did = obs_publisher_did
        fname_meta = RucioMeta.get_meta_fname(fname=minimal_fits_restored.path)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath_meta = os.path.join(tmpdir, os.path.split(fname_meta)[-1])
            _ = rm.to_dict(fpath=fpath_meta)
            assert os.path.exists(fpath_meta)
