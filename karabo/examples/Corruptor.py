# -*- coding: utf-8 -*-
"""Simulates visibilities using OSKAR and corrupts them.

Usage: python corruptor.py <oskar_sim_interferometer.ini>

The command line arguments are:
    - oskar_sim_interferometer.ini: Path to a settings file for the
                                    oskar_sim_interferometer app.
"""
from __future__ import print_function

import numpy as np
import oskar


class Corruptor(oskar.Interferometer):
    """Corrupts visibilities on-the-fly from OSKAR."""

    def __init__(self, precision=None, oskar_settings=None):
        oskar.Interferometer.__init__(self, precision, oskar_settings)

        # Do any other initialisation here...
        print("Initialising...")

    def finalise(self):
        """Called automatically by the base class at the end of run()."""
        oskar.Interferometer.finalise(self)

        # Do any other finalisation here...
        print("Finalising...")

    def process_block(self, block, block_index):
        """Corrupts the visibility block amplitude data.

        Args:
            block (oskar.VisBlock): The block to be processed.
            block_index (int):      The index of the visibility block.
        """
        # Get handles to visibility block data as numpy arrays.
        # uu = block.baseline_uu_metres()
        # vv = block.baseline_vv_metres()
        # ww = block.baseline_ww_metres()
        amp = block.cross_correlations()

        # Corrupt visibility amplitudes in the block here as needed
        # by messing with amp array.
        # uu, vv, ww have dimensions (num_times,num_baselines)
        # amp has dimensions (num_times,num_channels,num_baselines,num_pols)
        print(
            "Processing block {}/{} (time index {}-{})...".format(
                block_index + 1,
                self.num_vis_blocks,
                block.start_time_index,
                block.start_time_index + block.num_times - 1,
            )
        )

        # Simplest example:
        # amp *= 2.0

        (visHeader, visHandle) = oskar.VisHeader.read(
            "/home/jennifer/Documents/CSCS/Results/Tests/test_diluted_continous_002.vis"
        )
        visBlock = oskar.VisBlock.create_from_header(visHeader)
        print(visHeader.num_blocks)
        visBlock.read(visHeader, visHandle, block_index)
        vis = visBlock.cross_correlations()

        amp += vis[:, :, :, 0:1]
        print(vis.shape)

        print(amp.shape)

        # Write corrupted visibilities in the block to file(s).
        self.write_block(block, block_index)


def main():
    """Main function for visibility corruptor."""

    # Load the OSKAR settings INI file for the application.
    params = {
        "simulator": {"use_gpus": True},
        "observation": {
            "num_channels": 1,
            "start_frequency_hz": 1.4639e9,
            "frequency_inc_hz": 1e7,
            "phase_centre_ra_deg": 20,
            "phase_centre_dec_deg": -30,
            "num_time_steps": 10,
            "start_time_utc": "2000-03-20 12:06:39",
            "length": "03:05:00.000",
        },
        "telescope": {
            "input_directory": "/home/jennifer/Documents/SKA_Sara/meerkat.tm",
            "aperture_array/array_pattern/normalise": True,
            "normalise_beams_at_phase_centre": True,
            "pol_mode": "Scalar",
            "allow_station_beam_duplication": True,
            "station_type": "Isotropic beam",
            "gaussian_beam/fwhm_deg": 1.0 * np.sqrt(2),
            "gaussian_beam/ref_freq_hz": 1.4639e9
            # Mid-frequency in the redshift range
        },
        "interferometer": {
            "oskar_vis_filename": "/home/jennifer/Documents/SKAHIIM_Pipeline/result/"
            "Visibilities/test_corruptor.vis",
            "channel_bandwidth_hz": 0,  # 10000000,
            "time_average_sec": 8,
            "ignore_w_components": True,
        },
    }

    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(params)

    # Set up the corruptor and run it (see method, above).
    corruptor = Corruptor(oskar_settings=settings)
    corruptor.run()


if __name__ == "__main__":
    main()
