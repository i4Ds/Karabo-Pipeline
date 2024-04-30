import os
import tempfile
from datetime import datetime

import numpy as np

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_rascil import (
    RascilDirtyImager,
    RascilDirtyImagerConfig,
    RascilImageCleaner,
    RascilImageCleanerConfig,
)
from karabo.imaging.util import auto_choose_dirty_imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.test.conftest import TFiles


def test_image_circle(tobject: TFiles):
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)

    dirty_imager = auto_choose_dirty_imager(vis)
    dirty = dirty_imager.create_dirty_image(
        DirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        )
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
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)

    dirty_imager = auto_choose_dirty_imager(vis)
    dirty = dirty_imager.create_dirty_image(
        DirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dirty.write_to_file(os.path.join(tmpdir, "dirty.fits"), overwrite=True)
    dirty.plot(title="Dirty Image")


def test_dirty_image_resample(tobject: TFiles):
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)
    SHAPE = 2048

    dirty_imager = auto_choose_dirty_imager(vis)
    dirty = dirty_imager.create_dirty_image(
        DirtyImagerConfig(
            visibility=vis,
            imaging_npixel=SHAPE,
            imaging_cellsize=3.878509448876288e-05,
        )
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
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)

    dirty_imager = RascilDirtyImager()
    dirty = dirty_imager.create_dirty_image(
        DirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
            combine_across_frequencies=False,
        )
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
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)

    dirty_imager = auto_choose_dirty_imager(vis)
    dirty = dirty_imager.create_dirty_image(
        DirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            imaging_cellsize=3.878509448876288e-05,
        )
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
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)

    dirty_imager = RascilDirtyImager()
    dirty = dirty_imager.create_dirty_image(
        RascilDirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            # TODO Does a value of 10 radians make sense?
            imaging_cellsize=10,
            combine_across_frequencies=False,
            override_cellsize=True,
        )
    )

    header = dirty.header
    cdelt_overwrite_cellsize_false = header["CDELT1"]

    dirty = dirty_imager.create_dirty_image(
        RascilDirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            imaging_cellsize=1,
            combine_across_frequencies=False,
            override_cellsize=True,
        )
    )

    header = dirty.header
    cdelt_overwrite_cellsize_true = header["CDELT1"]

    assert cdelt_overwrite_cellsize_false == cdelt_overwrite_cellsize_true


def test_cellsize_overwrite_false(tobject: TFiles):
    vis = Visibility.read_from_file(tobject.visibilities_gleam_ms)
    dirty_imager = RascilDirtyImager()
    dirty = dirty_imager.create_dirty_image(
        RascilDirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            # TODO Does a value of 10 radians make sense?
            imaging_cellsize=10,
            combine_across_frequencies=False,
            override_cellsize=False,
        )
    )
    cdelt_overwrite_cellsize_false = dirty.header["CDELT1"]

    dirty = dirty_imager.create_dirty_image(
        RascilDirtyImagerConfig(
            visibility=vis,
            imaging_npixel=2048,
            imaging_cellsize=1,
            combine_across_frequencies=False,
            override_cellsize=False,
        )
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

    # could fail if `xarray` and `ska-sdp-func-python` not compatible, see issue #542
    (
        deconvolved,
        restored,
        residual,
    ) = RascilImageCleaner().create_cleaned_image_variants(
        RascilImageCleanerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            ms_file_path=visibility_askap.ms_file_path,
            ingest_vis_nchan=16,
            clean_nmajor=1,
            clean_algorithm="mmclean",
            clean_scales=[10, 30, 60],
            clean_threshold=0.12e-3,
            clean_nmoment=5,
            clean_psf_support=640,
            clean_restored_output="integrated",
            use_dask=True,
        )
    )

    assert os.path.exists(deconvolved.path)
    assert os.path.exists(restored.path)
    assert os.path.exists(residual.path)


# # TODO: move these on to CSCS Test Infrastructure once we have it.
# def test_clean():
#     sky = get_GLEAM_Sky()
#     phase_center = [250, -80]
#     sky = sky.filter_by_radius(0, .55, phase_center[0], phase_center[1])
#     sky.setup_default_wcs(phase_center=phase_center)
#     tel = Telescope.constructor("ASKAP")
#     observation_settings = Observation(100e6,
#                                        phase_centre_ra_deg=phase_center[0],
#                                        phase_centre_dec_deg=phase_center[1],
#                                        number_of_channels=64,
#                                        number_of_time_steps=24)
#
#     interferometer_sim = InterferometerSimulation(channel_bandwidth_hz=1e6)
#     visibility = interferometer_sim.run_simulation(tel, sky, observation_settings)
#
#     # visibility = Visibility()
#     # visibility.load_ms_file(f"./{data_path}/visibilities_gleam.ms")
#     imager = Imager(visibility,
#                     ingest_vis_nchan=16,
#                     ingest_chan_per_blockvis=1,
#                     ingest_average_blockvis=True,
#                     imaging_npixel=512,
#                     imaging_cellsize=3.878509448876288e-05*4,
#                     imaging_weighting='robust',
#                     imaging_robustness=-.5)
#     deconvoled_image, restored_image, residual_image = imager.imaging_rascil()
#     restored_image.save_to_file("result/restored.fits")
#     residual_image.save_to_file("result/residual.fits")
#     sky.save_to_file("result/imaging_sky.txt")

# def test_source_detection():
#     restored = open_fits_image("karabo/test/data/restored.fits")
#     detection_result = detect_sources_in_image(restored)
#     residual = detection_result.get_gaussian_residual_image()
#     residual.save_as_fits("result/gaus_residual.fits")
#     residual.plot()
#     detection_result.detection.show_fit()

# def test_source_detection_on_residual():
#     residual = open_fits_image("./data/residual.fits")
#     sources = detect_sources_in_image(residual, beam=(0.06, 0.02, 13.3))
#     print(sources.sources)

# def test_source_detection_on_dirty_image():
#     dirty = open_fits_image("./data/dirty.fits")
#     dirty.plot()
#     detection = detect_sources_in_image(dirty, beam=(0.06, 0.02, 13.3))
#     detection.get_island_mask().plot()

# def test_show_deconvolved():
#     deconv = open_fits_image("./data/deconvolved.fits")
#     deconv.plot()

# def test_try_star_finder():
#     deconvolved = open_fits_image("./data/deconvolved.fits")
#     use_dao_star_finder(deconvolved)
#
# def test_source_detection_on_deconvolved():
#     deconvolved = open_fits_image("./data/deconvolved.fits")
# sources = detect_sources_in_image(
#     deconvolved,
#     # beam=(0.05425680530067645, 0.05047422298502832, -205.3215315790252))
#     beam=(0.06, 0.02, 13.3),
# )
#     print(sources.sources)
