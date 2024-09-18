import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.util import auto_choose_dirty_imager_from_vis
from karabo.simulation.beam import BeamPattern
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import ObservationLong
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.test.conftest import TFiles


# Test cases
def test_fit_element(tobject: TFiles):
    tel = Telescope.constructor("MeerKAT")
    beam = BeamPattern(tobject.run5_cst)
    beam.fit_elements(tel, freq_hz=1.0e08, avg_frac_error=0.5)


def test_katbeam():
    beampixels = BeamPattern.get_meerkat_uhfbeam(
        f=800, pol="I", beamextentx=40, beamextenty=40
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        BeamPattern.show_kat_beam(
            beampixels[0],
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


def test_long_observations(tobject: TFiles, sky_data: NDArray[np.float64]):
    # skips `input` during unit tests if using `karabo.util.data_util.input_wrapper`
    os.environ["SKIP_INPUT"] = str(True)
    number_of_days = 3
    hours_per_day = 4
    enable_array_beam = False
    with tempfile.TemporaryDirectory() as tmpdir:
        combined_ms_filepath = os.path.join(tmpdir, "combined_vis.ms")
        sky = SkyModel()
        sky.add_point_sources(sky_data)
        telescope = Telescope.constructor("MeerKAT")
        observation_long = ObservationLong(
            mode="Tracking",
            phase_centre_ra_deg=20.0,
            start_date_and_time=datetime(2000, 1, 1, 11, 00, 00, 521489),
            length=timedelta(hours=hours_per_day, minutes=0, seconds=0, milliseconds=0),
            phase_centre_dec_deg=-30.0,
            number_of_time_steps=7,
            start_frequency_hz=1.0e9,
            frequency_increment_hz=1e6,
            number_of_channels=3,
            number_of_days=number_of_days,
        )
        beam_polX = BeamPattern(
            cst_file_path=tobject.cst_like_beam_port_1_txt,
            telescope=telescope,
            freq_hz=observation_long.start_frequency_hz,
            pol="X",
            avg_frac_error=0.001,
            beam_method="Gaussian Beam",
        )
        beam_polY = BeamPattern(
            cst_file_path=tobject.cst_like_beam_port_2_txt,
            telescope=telescope,
            freq_hz=observation_long.start_frequency_hz,
            pol="Y",
            avg_frac_error=0.001,
            beam_method="Gaussian Beam",
        )
        simulation = InterferometerSimulation(
            ms_file_path=combined_ms_filepath,
            channel_bandwidth_hz=2e7,
            time_average_sec=7,
            noise_enable=False,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=1.0e9,
            noise_inc_freq=1.0e6,
            noise_number_freq=1,
            noise_rms_start=0.1,
            noise_rms_end=1,
            enable_numerical_beam=enable_array_beam,
            enable_array_beam=enable_array_beam,
            beam_polX=beam_polX,
            beam_polY=beam_polY,
        )
        # -------- Iterate over days
        visibility = simulation.run_simulation(
            telescope=telescope,
            sky=sky,
            observation=observation_long,
        )

        dirty_imager = auto_choose_dirty_imager_from_vis(
            visibility,
            DirtyImagerConfig(
                imaging_npixel=4096,
                imaging_cellsize=1.0e-5,
            ),
        )
        dirty_imager.create_dirty_image(visibility)
