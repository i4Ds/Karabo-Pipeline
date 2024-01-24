import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest
from numpy.typing import NDArray

from karabo.imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.visibility import Visibility
from karabo.util.data_util import Gauss, resample_spectral_lines


def simulate_spectral_vis(
    ra_spec,
    dec_spec,
    phase_ra,
    phase_dec,
    spectral_freq0,
    npoints,
    dfreq,
    spec_line,
    spec_path,
    bandwidth_smearing,
    chan_width,
):
    """
    Simulates spectral line point sources
    """
    dfreq_sampled, line_sampled = resample_spectral_lines(npoints, dfreq, spec_line)
    spectral_sky_data = np.zeros((npoints, 12))
    spectral_sky_data[:, 0] = ra_spec
    spectral_sky_data[:, 1] = dec_spec
    spectral_vis_output = [0] * npoints
    spectral_ms_output = [0] * npoints
    freq_spec = [0] * npoints
    for i in range(npoints):
        spectral_vis_output[i] = f"{spec_path}_vis_spectral_{i}.vis"
        spectral_ms_output[i] = f"{spec_path}_vis_spectral_{i}.ms"
        os.system("rm -rf " + spectral_vis_output[i])
        os.system("rm -rf " + spectral_ms_output[i])
        print(
            "#### Computing Visibilities for Spectral Lines Begins for "
            + str(i + 1)
            + "/"
            + str(npoints)
        )
        freq_spec[i] = spectral_freq0 + dfreq_sampled[i]
        spectral_sky = SkyModel()
        telescope = Telescope.constructor("MeerKAT")
        spectral_sky_data[i, 2] = line_sampled[i]
        spectral_sky_data[i, 6] = freq_spec[i]
        spectral_sky.add_point_sources(spectral_sky_data)
        simulation = InterferometerSimulation(
            vis_path=spectral_vis_output[i],
            channel_bandwidth_hz=bandwidth_smearing,
            time_average_sec=1,
            noise_enable=False,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=1e9,
            noise_inc_freq=1e8,
            noise_number_freq=24,
            noise_rms_start=5000,
            noise_rms_end=10000,
        )
        observation = Observation(
            phase_centre_ra_deg=phase_ra,
            phase_centre_dec_deg=phase_dec,
            start_date_and_time=datetime(2022, 1, 1, 23, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            number_of_time_steps=1,
            start_frequency_hz=freq_spec[i],
            frequency_increment_hz=chan_width,
            number_of_channels=1,
        )
        visibility = simulation.run_simulation(telescope, spectral_sky, observation)
        visibility.write_to_file(spectral_ms_output[i])
    return spectral_vis_output, spectral_ms_output


@pytest.mark.skip(reason="faulty test (or karabo-code?), needs to be adapted.")
def test_disabled_spectral_line(sky_data: NDArray[np.float64]):
    # ------- Simulate Foreground ---------#
    make_foreground_image = 0
    bandwidth_smearing = 0
    phase_ra = 20.0
    phase_dec = -30.0
    obs_freq = 0.99e9  # 999.0 MHz & 0.2 MHz
    chan_width = 1e6  # 100 kHz
    with tempfile.TemporaryDirectory() as tmpdir:
        foreground_vis_file = os.path.join(tmpdir, "vis_foreground.vis")
        foreground_ms_file = os.path.join(tmpdir, "vis_foreground.ms")
        write_foreground_ms = True
        foreground = SkyModel()
        foreground.add_point_sources(sky_data)
        telescope = Telescope.constructor("MeerKAT")
        simulation = InterferometerSimulation(
            vis_path=foreground_vis_file,
            channel_bandwidth_hz=bandwidth_smearing,
            time_average_sec=1,
            noise_enable=False,
            noise_seed="time",
            noise_freq="Range",
            noise_rms="Range",
            noise_start_freq=1.0e9,
            noise_inc_freq=1.0e8,
            noise_number_freq=24,
            noise_rms_start=5000,
            noise_rms_end=10000,
        )
        foreground_observation = Observation(
            phase_centre_ra_deg=phase_ra,
            phase_centre_dec_deg=phase_dec,
            start_date_and_time=datetime(2022, 1, 1, 23, 00, 00, 521489),
            length=timedelta(hours=0, minutes=0, seconds=1, milliseconds=0),
            number_of_time_steps=1,
            start_frequency_hz=obs_freq,
            frequency_increment_hz=chan_width,
            number_of_channels=20,
        )
        (
            visibility,
            foreground_cross_correlation,
            fg_header,
            fg_handle,
            fg_block,
            ff_uu,
            ff_vv,
            ff_ww,
        ) = simulation.simulate_foreground_vis(
            telescope,
            foreground,
            foreground_observation,
            foreground_vis_file,
            write_foreground_ms,
            foreground_ms_file,
        )
        if make_foreground_image:
            imager = Imager(
                visibility, imaging_npixel=2048 * 1, imaging_cellsize=50
            )  # imaging cellsize is over-written in the Imager based on max uv dist.
            dirty = imager.get_dirty_image()
            dirty.write_to_file(os.path.join(tmpdir, "foreground.fits"), overwrite=True)
            dirty.plot(title="Flux Density (Jy)")
        # ------- Simulate Spectral Line Sky -----#
        spectral_freq0 = 1.0e9
        make_spectral_image = 0
        dfreq = np.linspace(-5, 5, 1000) * 1.0e6
        spec_line = Gauss(dfreq, 0, 0, 10, 2.0e6)
        npoints = int((dfreq[-1] - dfreq[0]) / chan_width)
        ra_spec = 20.2
        dec_spec = -30.2
        spectral_vis_output, _ = simulate_spectral_vis(
            ra_spec,
            dec_spec,
            phase_ra,
            phase_dec,
            spectral_freq0,
            npoints,
            dfreq,
            spec_line,
            tmpdir,
            bandwidth_smearing,
            chan_width,
        )
        if make_spectral_image:
            imager = Imager(
                visibility, imaging_npixel=2048 * 1, imaging_cellsize=50
            )  # imaging cellsize is over-written in the Imager based on max uv dist.
            dirty = imager.get_dirty_image()
            dirty.write_to_file(
                os.path.join(tmpdir, "foreground.fits"),
                overwrite=True,
            )
            dirty.plot(title="Flux Density (Jy)")

        # ------- Combine Visibilities ----------#
        dfreq = np.linspace(-5, 5, 1000) * 1.0e6
        chan_width = 1e6
        npoints = int((dfreq[-1] - dfreq[0]) / chan_width)
        foreground_vis_file = os.path.join(tmpdir, "vis_foreground.vis")
        combined_vis_filepath = os.path.join(tmpdir, "combined_fg_spectral_line_vis.ms")
        spectral_vis_output = [0] * npoints
        for i in range(npoints):
            spectral_vis_output[i] = os.path.join(tmpdir, f"vis_spectral_{i}.vis")
        Visibility.combine_spectral_foreground_vis(
            foreground_vis_file, spectral_vis_output, combined_vis_filepath
        )
