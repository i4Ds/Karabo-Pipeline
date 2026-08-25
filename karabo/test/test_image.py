import math
import os
import tempfile
from datetime import datetime

import numpy as np

from karabo.imaging.backends.sdp_backend import SdpImager, SdpImagerConfig
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_interface import ImageSpec
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles
from karabo.test.util import create_compatible_dirty_image


def _create_sdp_dirty_image(
    visibility: Visibility,
    npixel: int,
    cellsize_radians: float,
    *,
    override_cellsize: bool = False,
):
    imager = SdpImager(
        SdpImagerConfig(
            combine_across_frequencies=False,
            override_cellsize=override_cellsize,
        )
    )
    dirty, _ = imager.invert(
        visibility,
        ImageSpec(
            npix=npixel,
            cellsize_arcsec=np.rad2deg(cellsize_radians) * 3600.0,
            phase_centre_deg=(0.0, 0.0),
        ),
    )
    return dirty


def test_image_circle(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty = create_compatible_dirty_image(
        vis,
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )

    data = dirty.data[0][0]  # Returns a 2D array, with values for each (x, y) pixel

    assert not np.any(np.isnan(data))

    # Apply in-place circle transformation, keeping only data within a circle
    dirty.circle()
    data = dirty.data[0][0]
    len_x, len_y = data.shape

    assert np.isnan(data[0][0])
    assert not np.isnan(data[len_x // 2][len_y // 2])


def test_dirty_image(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty = create_compatible_dirty_image(
        vis,
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dirty.write_to_file(os.path.join(tmpdir, "dirty.fits"), overwrite=True)
    dirty.plot(title="Dirty Image")


def test_dirty_image_resample(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)
    SHAPE = 2048

    dirty = create_compatible_dirty_image(
        vis,
        DirtyImagerConfig(
            imaging_npixel=SHAPE,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )

    shape_before = dirty.data.shape
    NEW_SHAPE = 512
    dirty.resample((NEW_SHAPE, NEW_SHAPE))
    with tempfile.TemporaryDirectory() as tmpdir:
        dirty.write_to_file(os.path.join(tmpdir, "dirty_resample.fits"), overwrite=True)
    dirty.plot(title="Dirty Image")

    assert dirty.data.shape[2] == NEW_SHAPE
    assert dirty.data.shape[3] == NEW_SHAPE
    assert dirty.data.shape[0] == shape_before[0]
    assert dirty.data.shape[1] == shape_before[1]
    assert np.sum(np.isnan(dirty.data)) == 0

    dirty.resample((SHAPE, SHAPE))

    assert dirty.data.shape[2] == SHAPE
    assert dirty.data.shape[3] == SHAPE
    assert dirty.data.shape[0] == shape_before[0]
    assert dirty.data.shape[1] == shape_before[1]
    assert np.sum(np.isnan(dirty.data)) == 0


def test_dirty_image_cutout(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty = _create_sdp_dirty_image(
        vis,
        npixel=2048,
        cellsize_radians=3.878509448876288e-05,
    )

    cutout1 = dirty.cutout((1000, 1000), (500, 500))

    assert cutout1.data.shape[2] == 500
    assert cutout1.data.shape[3] == 500
    assert cutout1.header["CRPIX1"] == 275  # Don't understand why but this is the value
    assert cutout1.header["CRPIX2"] == 275
    assert cutout1.header["CRVAL1"] == 250
    assert cutout1.header["CRVAL2"] == -80

    assert np.sum(np.isnan(cutout1.data)) == 0
    assert np.all(
        np.equal(cutout1.data[0, 0, :, :], dirty.data[0, 0, 750:1250, 750:1250])
    )


def test_dirty_image_N_cutout(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty = create_compatible_dirty_image(
        vis,
        DirtyImagerConfig(
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        ),
    )

    cutouts = dirty.split_image(N=4)

    assert len(cutouts) == 16

    for cutout in cutouts:
        assert cutout.data.shape[2] == 512
        assert cutout.data.shape[3] == 512
        assert np.sum(np.isnan(cutout.data)) == 0

    cutouts = dirty.split_image(N=2, overlap=50)

    assert len(cutouts) == 4

    for cutout in cutouts:
        assert cutout.data.shape[2] == 1024 + 50
        assert cutout.data.shape[3] == 1024 + 50
        assert np.sum(np.isnan(cutout.data)) == 0


def test_cellsize_overwrite(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)

    dirty = _create_sdp_dirty_image(
        vis, npixel=256, cellsize_radians=10, override_cellsize=True
    )

    header = dirty.header
    cdelt_overwrite_cellsize_false = header["CDELT1"]

    dirty = _create_sdp_dirty_image(
        vis, npixel=256, cellsize_radians=1, override_cellsize=True
    )

    header = dirty.header
    cdelt_overwrite_cellsize_true = header["CDELT1"]

    assert cdelt_overwrite_cellsize_false == cdelt_overwrite_cellsize_true


def test_cellsize_overwrite_false(tobject: TFiles):
    vis = Visibility(tobject.visibilities_gleam_ms)
    dirty = _create_sdp_dirty_image(
        vis, npixel=256, cellsize_radians=10, override_cellsize=False
    )
    cdelt_overwrite_cellsize_false = dirty.header["CDELT1"]

    dirty = _create_sdp_dirty_image(
        vis, npixel=256, cellsize_radians=1, override_cellsize=False
    )
    cdelt_overwrite_cellsize_true = dirty.header["CDELT1"]

    assert cdelt_overwrite_cellsize_false != cdelt_overwrite_cellsize_true


def test_imaging():
    phase_center = [250, -80]
    gleam_sky = SkyModel.get_GLEAM_Sky(min_freq=72e6, max_freq=80e6)
    sky = gleam_sky.filter_by_radius(0, 0.55, phase_center[0], phase_center[1])
    sky.setup_default_wcs(phase_center=phase_center)
    askap_tel = Telescope.constructor("ASKAP")
    observation_settings = Observation(
        start_frequency_hz=100e6,
        start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
        phase_centre_ra_deg=phase_center[0],
        phase_centre_dec_deg=phase_center[1],
        number_of_channels=16,
        number_of_time_steps=24,
    )

    interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
    visibility_askap = interferometer_sim.run_simulation(
        askap_tel,
        sky,
        observation_settings,
    )
    imaging_npixel = 2048
    imaging_cellsize = 3.878509448876288e-05

    imager = SdpImager(
        SdpImagerConfig(
            combine_across_frequencies=False,
            clean_algorithm="hogbom",
            clean_threshold=0.12e-3,
        )
    )
    dirty, psf = imager.invert(
        visibility_askap,
        ImageSpec(
            npix=imaging_npixel,
            cellsize_arcsec=math.degrees(imaging_cellsize) * 3600.0,
            phase_centre_deg=(phase_center[0], phase_center[1]),
        ),
    )
    restored = imager.restore(dirty, psf)
    deconvolved = imager.last_model_image
    residual = imager.last_residual_image

    assert os.path.exists(deconvolved.path)
    assert os.path.exists(restored.path)
    assert os.path.exists(residual.path)
