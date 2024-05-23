import os
import tempfile
from datetime import datetime, timedelta

from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.simulation.beam import BeamPattern
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.test.conftest import TFiles


def test_fit_element(tobject: TFiles):
    tel = Telescope.constructor("MeerKAT")
    beam = BeamPattern(tobject.run5_cst)
    beam.fit_elements(tel, freq_hz=1e8, avg_frac_error=0.5)


def test_katbeam():
    beampixels = BeamPattern.get_meerkat_uhfbeam(
        f=800, pol="I", beamextentx=40, beamextenty=40
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        BeamPattern.show_kat_beam(
            beampixels[2],
            40,
            800,
            "I",
            path=os.path.join(tmpdir, "katbeam_beam.png"),
        )


def test_eidosbeam():
    npix = 500
    dia = 10
    thres = 0
    ch = 0
    B_ah = BeamPattern.get_eidos_holographic_beam(
        npix=npix, ch=ch, dia=dia, thres=thres, mode="AH"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        BeamPattern.show_eidos_beam(
            B_ah, path=os.path.join(tmpdir, "eidos_AH_beam.png")
        )
        B_em = BeamPattern.get_eidos_holographic_beam(
            npix=npix, ch=ch, dia=dia, thres=thres, mode="EM"
        )
        BeamPattern.show_eidos_beam(
            B_em, path=os.path.join(tmpdir, "eidos_EM_beam.png")
        )
        BeamPattern.eidos_lineplot(
            B_ah, B_em, npix, path=os.path.join(tmpdir, "eidos_residual_beam.png")
        )


def test_gaussian_beam():
    """
    We test that image reconstruction works also with a Gaussian beam and
    test both Imagers: Oskar and Rascil.
    """
    # --------------------------
    freq = 8.0e8
    precision = "double"
    beam_type = "Gaussian beam"

    sky = SkyModel.sky_test()

    telescope = Telescope.constructor("MeerKAT")
    # Remove beam if already present
    test = os.listdir(telescope.path)
    for item in test:
        if item.endswith(".bin"):
            os.remove(os.path.join(telescope.path, item))
    # ------------- Simulation Begins
    with tempfile.TemporaryDirectory() as tmpdir:
        simulation = InterferometerSimulation(
            vis_path=os.path.join(tmpdir, "beam_vis.vis"),
            ms_file_path=os.path.join(tmpdir, "beam_vis.ms"),
            channel_bandwidth_hz=2e7,
            time_average_sec=8,
            noise_enable=False,
            ignore_w_components=True,
            precision=precision,
            use_gpus=False,
            station_type=beam_type,
            gauss_beam_fwhm_deg=1.0,
            gauss_ref_freq_hz=1.5e9,
        )
        observation = Observation(
            mode="Tracking",
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2000, 3, 20, 12, 6, 39, 0),
            length=timedelta(hours=3, minutes=5, seconds=0, milliseconds=0),
            phase_centre_dec_deg=-30.0,
            number_of_time_steps=10,
            start_frequency_hz=freq,
            frequency_increment_hz=2e7,
            number_of_channels=1,
        )
        visibility = simulation.run_simulation(telescope, sky, observation)

        # RASCIL IMAGING
        dirty_imager = RascilDirtyImager(
            RascilDirtyImagerConfig(
                imaging_npixel=4096,
                imaging_cellsize=2.13e-5,
            )
        )
        dirty = dirty_imager.create_dirty_image(visibility)
        dirty.plot(title="Flux Density (Jy)")
