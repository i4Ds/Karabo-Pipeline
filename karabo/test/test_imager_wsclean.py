import math
import os
import shlex
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import karabo.imaging.imager_wsclean as wsclean_module
from karabo.imaging.backends.wsclean_backend import WscleanBackendImager
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_factory import (
    ImagingBackend,
    WscleanBackendConfig,
    get_imager,
)
from karabo.imaging.imager_interface import ImageSpec
from karabo.imaging.imager_wsclean import (
    WscleanDirtyImager,
    WscleanImageCleaner,
    WscleanImageCleanerConfig,
    create_image_custom_command,
)
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles
from karabo.util.file_handler import FileHandler
from karabo.warning import DIRECT_WSCLEAN_USAGE_MESSAGE, DirectWscleanUsageWarning

# this parameter sets the number of interations that WSClean does
# to clean an image. The default is set to 50_000. We don't need
# that many because we do run tests only, i.e. the quality of the
# result is not tested.
CLEAN_ITERATIONS = 100
WSCLEAN_AVAILABLE = shutil.which("wsclean") is not None


def test_wsclean_imager_factory_returns_adapter():
    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        imager = get_imager(ImagingBackend.WSCLEAN)
    assert not any(
        isinstance(warning.message, DirectWscleanUsageWarning)
        for warning in warning_record
    )
    assert isinstance(imager, WscleanBackendImager)


def test_wsclean_imager_factory_forwards_matching_config():
    config = WscleanBackendConfig(weighting="natural", clean_niter=23)
    imager = get_imager(ImagingBackend.WSCLEAN, config=config)

    assert isinstance(imager, WscleanBackendImager)
    assert imager.config is config


def test_wsclean_backend_config_rejects_unsupported_weighting():
    with pytest.raises(ValueError, match="weighting must be"):
        WscleanBackendConfig(weighting="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options, message",
    [
        ({"weighting": "briggs"}, "requires an explicit robustness"),
        ({"robustness": 0.0}, "requires weighting='briggs'"),
        ({"weighting": "natural", "robustness": 0.5}, "requires weighting='briggs'"),
        ({"weighting": "uniform", "robustness": 0.5}, "requires weighting='briggs'"),
        ({"weighting": "briggs", "robustness": "0.5"}, "finite real number"),
        ({"weighting": "briggs", "robustness": True}, "finite real number"),
        ({"weighting_rank_filter": -1.0}, "non-negative"),
        ({"weighting_rank_filter": True}, "finite real number"),
        ({"weighting_rank_filter_size": 0}, "positive integer"),
        ({"weighting_rank_filter_size": -1}, "positive integer"),
        ({"weighting_rank_filter_size": 2.5}, "positive integer"),
        ({"weighting_rank_filter_size": True}, "positive integer"),
        ({"taper_gaussian_arcsec": 0.0}, "positive"),
        ({"taper_gaussian_arcsec": -1.0}, "positive"),
        ({"taper_gaussian_arcsec": "8asec"}, "finite real number"),
    ],
)
def test_wsclean_backend_config_rejects_invalid_options(options, message):
    with pytest.raises(ValueError, match=message):
        WscleanBackendConfig(**options)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "name", ["robustness", "weighting_rank_filter", "taper_gaussian_arcsec"]
)
def test_wsclean_backend_config_rejects_nonfinite_options(name, value):
    options = {name: value}
    if name == "robustness":
        options["weighting"] = "briggs"
    with pytest.raises(ValueError, match="finite real number"):
        WscleanBackendConfig(**options)


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, []),
        ({"weighting": "natural"}, ["-weight", "natural"]),
        ({"weighting": "uniform"}, ["-weight", "uniform"]),
        ({"weighting": "briggs", "robustness": 0.0}, ["-weight", "briggs", "0.0"]),
        ({"weighting_rank_filter": 0.0}, ["-weighting-rank-filter", "0.0"]),
        ({"weighting_rank_filter_size": 64}, ["-weighting-rank-filter-size", "64"]),
        ({"taper_gaussian_arcsec": 8.0}, ["-taper-gaussian", "8.0asec"]),
        (
            {
                "weighting": "briggs",
                "robustness": -0.5,
                "weighting_rank_filter": 3.0,
                "weighting_rank_filter_size": 32,
                "taper_gaussian_arcsec": 8.0,
            },
            [
                "-weight",
                "briggs",
                "-0.5",
                "-weighting-rank-filter",
                "3.0",
                "-weighting-rank-filter-size",
                "32",
                "-taper-gaussian",
                "8.0asec",
            ],
        ),
    ],
)
def test_wsclean_common_arguments_reach_all_commands(
    monkeypatch, tmp_path, options, expected
):
    commands = []

    def capture_command(command, **kwargs):
        argv = shlex.split(command)
        commands.append(argv[argv.index("wsclean") + 1 :])
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(wsclean_module.subprocess, "run", capture_command)
    monkeypatch.setattr(wsclean_module, "Image", SimpleNamespace)
    monkeypatch.setattr(wsclean_module.shutil, "copyfile", lambda *args: None)
    monkeypatch.setattr(
        FileHandler, "get_tmp_dir", lambda *args, **kwargs: str(tmp_path)
    )
    vis = SimpleNamespace(format="MS", path="/tmp/input.ms")
    imager = get_imager(ImagingBackend.WSCLEAN, config=WscleanBackendConfig(**options))
    spec = ImageSpec(npix=64, cellsize_arcsec=2.0, phase_centre_deg=(0.0, 0.0))
    dirty, psf = imager.invert(vis, spec)
    imager.restore(dirty, psf)

    common = expected + [
        "-size",
        "64",
        "64",
        "-scale",
        f"{math.degrees(spec.cellsize_radians)}deg",
    ]
    assert commands == [
        common + [vis.path],
        common + ["-make-psf", vis.path],
        common
        + [
            "-reuse-dirty",
            "pre_existing",
            "-niter",
            "100",
            "-mgain",
            "0.8",
            "-auto-threshold",
            "3",
            vis.path,
        ],
    ]


def test_direct_wsclean_dirty_imager_warns():
    with pytest.warns(DirectWscleanUsageWarning) as warning_record:
        WscleanDirtyImager(
            DirtyImagerConfig(
                imaging_npixel=64,
                imaging_cellsize=3.878509448876288e-05,
            ),
        )
    assert str(warning_record[0].message) == DIRECT_WSCLEAN_USAGE_MESSAGE


def test_direct_wsclean_cleaner_warns():
    with pytest.warns(DirectWscleanUsageWarning) as warning_record:
        WscleanImageCleaner(
            WscleanImageCleanerConfig(
                imaging_npixel=64,
                imaging_cellsize=3.878509448876288e-05,
                niter=CLEAN_ITERATIONS,
            ),
        )
    assert str(warning_record[0].message) == DIRECT_WSCLEAN_USAGE_MESSAGE


def test_direct_wsclean_custom_command_warns():
    with pytest.warns(DirectWscleanUsageWarning) as warning_record:
        with pytest.raises(ValueError):
            create_image_custom_command("not-wsclean")
    assert str(warning_record[0].message) == DIRECT_WSCLEAN_USAGE_MESSAGE


def test_wsclean_restore_before_invert_raises():
    imager = get_imager(ImagingBackend.WSCLEAN)
    with pytest.raises(RuntimeError, match="previous invert"):
        imager.restore(None, None)  # type: ignore[arg-type]


@pytest.mark.skipif(not WSCLEAN_AVAILABLE, reason="WSClean is not installed")
@pytest.mark.parametrize(
    "config",
    [
        WscleanBackendConfig(),
        WscleanBackendConfig(weighting="briggs", robustness=0.5),
        WscleanBackendConfig(taper_gaussian_arcsec=60.0),
        WscleanBackendConfig(
            weighting="briggs",
            robustness=-0.5,
            weighting_rank_filter=3.0,
            weighting_rank_filter_size=32,
            taper_gaussian_arcsec=60.0,
        ),
    ],
    ids=["defaults", "briggs", "gaussian-taper", "combined"],
)
def test_wsclean_imager_factory_invert_and_restore_smoke(
    minimal_casa_ms: Visibility,
    config: WscleanBackendConfig,
):
    imager = get_imager(ImagingBackend.WSCLEAN, config=config)
    spec = ImageSpec(
        npix=64,
        cellsize_arcsec=math.degrees(5e-5) * 3600.0,
        phase_centre_deg=(0.0, 0.0),
    )

    dirty_image, psf_image = imager.invert(minimal_casa_ms, spec)

    assert os.path.exists(dirty_image.path)
    assert os.path.exists(psf_image.path)
    assert dirty_image.data.size > 0
    assert psf_image.data.size > 0

    restored = imager.restore(dirty_image, psf_image)
    assert os.path.exists(restored.path)
    assert restored.data.size > 0
    for image in (dirty_image, psf_image, restored):
        data = image.get_squeezed_data()
        assert data.shape == (64, 64)
        assert np.isfinite(data).all()
        assert np.max(np.abs(data)) > 0
        assert image.header["CTYPE1"].startswith("RA")
        assert image.header["CTYPE2"].startswith("DEC")
    assert restored.header["BMAJ"] > 0
    assert restored.header["BMIN"] > 0


def test_dirty_image(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(vis)

    assert os.path.exists(dirty_image.path)


def test_dirty_image_custom_path(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)
    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )
    with FileHandler() as tmp_dir:
        output_fits_path = os.path.join(
            tmp_dir,
            "test_dirty_image_custom_path.fits",
        )
        dirty_image = dirty_imager.create_dirty_image(
            vis,
            output_fits_path=output_fits_path,
        )

        assert dirty_image.path == output_fits_path
        assert os.path.exists(dirty_image.path)


def test_create_cleaned_image(default_sample_simulation_visibility: Visibility):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            niter=CLEAN_ITERATIONS,
        )
    ).create_cleaned_image(default_sample_simulation_visibility)

    assert os.path.exists(restored.path)


def test_create_cleaned_image_custom_path(
    default_sample_simulation_visibility: Visibility,
):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    with FileHandler() as tmp_dir:
        output_fits_path = os.path.join(
            tmp_dir,
            "test_create_cleaned_image_custom_path.fits",
        )
        restored = WscleanImageCleaner(
            WscleanImageCleanerConfig(
                imaging_npixel=imaging_npixel,
                imaging_cellsize=imaging_cellsize,
                niter=CLEAN_ITERATIONS,
            )
        ).create_cleaned_image(
            default_sample_simulation_visibility,
            output_fits_path=output_fits_path,
        )

        assert restored.path == output_fits_path
        assert os.path.exists(restored.path)


def test_create_cleaned_image_reuse_dirty(
    default_sample_simulation_visibility: Visibility,
):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    dirty_imager = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
        ),
    )
    dirty_image = dirty_imager.create_dirty_image(default_sample_simulation_visibility)
    assert os.path.exists(dirty_image.path)

    restored = WscleanImageCleaner(
        WscleanImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            niter=CLEAN_ITERATIONS,
        )
    ).create_cleaned_image(
        default_sample_simulation_visibility,
        dirty_fits_path=dirty_image.path,
    )
    assert os.path.exists(dirty_image.path)
    assert os.path.exists(restored.path)


def test_create_image_custom_command(default_sample_simulation_visibility: Visibility):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored = create_image_custom_command(
        "wsclean "
        f"-size {imaging_npixel} {imaging_npixel} "
        f"-scale {math.degrees(imaging_cellsize)}deg "
        f"-niter {CLEAN_ITERATIONS} "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{default_sample_simulation_visibility.path}",
    )

    assert os.path.exists(restored.path)


def test_create_image_custom_command_multiple_outputs(
    default_sample_simulation_visibility: Visibility,
):
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    restored, residual = create_image_custom_command(
        "wsclean "
        f"-size {imaging_npixel} {imaging_npixel} "
        f"-scale {math.degrees(imaging_cellsize)}deg "
        f"-niter {CLEAN_ITERATIONS} "
        "-mgain 0.8 "
        "-auto-threshold 3 "
        f"{default_sample_simulation_visibility.path}",
        ["wsclean-image.fits", "wsclean-residual.fits"],
    )  # type: ignore  # cannot infer type, should be List[Image]

    assert os.path.exists(restored.path)
    assert os.path.exists(residual.path)
