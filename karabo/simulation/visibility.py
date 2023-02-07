import os
import os.path
import shutil

import numpy as np
import oskar

from karabo.karabo_resource import KaraboResource
from karabo.util.FileHandle import FileHandle


class Visibility(KaraboResource):
    def __init__(self):
        self.file = FileHandle(is_dir=True, suffix=".ms")

    def write_to_file(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
        shutil.copytree(self.file.path, path)

    @staticmethod
    def read_from_file(path: str) -> any:
        file = FileHandle(path, is_dir=True)
        vis = Visibility()
        vis.file = file
        return vis

    def combine_spectral_foreground_vis(
        foreground_vis_file, spectral_vis_output, combined_vis_filepath
    ):
        """
        This function combines the visibilities of foreground and spectral lines
        Inputs: foreground visibility file, list of spectral line vis files, output path & name of combined vis file
        """
        print("#--- Performing visibilities combination...")
        (fg_header, fg_handle) = oskar.VisHeader.read(foreground_vis_file)
        foreground_cross_correlation = [0] * fg_header.num_blocks
        fg_chan = [0] * fg_header.num_blocks
        foreground_freq = fg_header.freq_start_hz + fg_header.freq_inc_hz * np.arange(
            fg_header.num_channels_total
        )
        nvis = len(spectral_vis_output)
        out_vis = [0] * nvis
        # fg_max_channel=fg_header.max_channels_per_block;
        for i in range(fg_header.num_blocks):
            fg_block = oskar.VisBlock.create_from_header(fg_header)
            fg_block.read(fg_header, fg_handle, i)
            fg_chan[i] = fg_block.num_channels
            foreground_cross_correlation[i] = fg_block.cross_correlations()
        ff_uu = fg_block.baseline_uu_metres()
        ff_vv = fg_block.baseline_vv_metres()
        ff_ww = fg_block.baseline_ww_metres()
        # for j in tqdm(range(nvis)):
        for j in range(nvis):
            (sp_header, sp_handle) = oskar.VisHeader.read(spectral_vis_output[j])
            spec_freq = sp_header.freq_start_hz
            spec_idx = int(
                np.where(
                    abs(foreground_freq - spec_freq)
                    == np.min(abs(foreground_freq - spec_freq))
                )[0]
            )
            print(spec_freq, spec_idx)
            out_vis[j] = [0] * sp_header.num_blocks
            for k in range(sp_header.num_blocks):
                sp_block = oskar.VisBlock.create_from_header(sp_header)
                sp_block.read(sp_header, sp_handle, k)
                out_vis[j][k] = sp_block.cross_correlations()
            block_num = int(spec_idx / fg_header.max_channels_per_block)
            chan_block_num = int(
                spec_idx - block_num * fg_header.max_channels_per_block
            )
            fcc = foreground_cross_correlation[block_num][:, chan_block_num, :, :]
            foreground_cross_correlation[block_num][:, chan_block_num, :, :] = (
                fcc + out_vis[j][0][0]
            )
        # --------- Writing the Visibilities
        os.system("rm -rf " + combined_vis_filepath)
        ms = oskar.MeasurementSet.create(
            combined_vis_filepath,
            fg_block.num_stations,
            fg_header.num_channels_total,
            fg_block.num_pols,
            fg_header.freq_start_hz,
            fg_header.freq_inc_hz,
        )
        ms.set_phase_centre(
            fg_header.phase_centre_ra_deg, fg_header.phase_centre_dec_deg
        )
        # Write data one block at a time.
        time_inc = fg_header.time_inc_sec
        print("### Writing combined visibilities in ", combined_vis_filepath)
        start_row = 0
        exposure_sec = fg_header.get_time_average_sec()
        sc = [0] * len(foreground_cross_correlation)
        fg_chan = [0] * len(foreground_cross_correlation)
        time_stamp = fg_header.get_time_start_mjd_utc()
        ms.write_coords(
            start_row,
            fg_block.num_baselines,
            ff_uu[0],
            ff_vv[0],
            ff_ww[0],
            exposure_sec,
            time_inc,
            time_stamp,
        )
        fcc_array = foreground_cross_correlation[0]
        for k in range(len(foreground_cross_correlation) - 1):
            fcc_array = np.hstack((fcc_array, foreground_cross_correlation[k + 1]))
        ms.write_vis(
            start_row,
            0,
            fg_header.num_channels_total,
            fg_block.num_baselines,
            fcc_array,
        )

    def simulate_foreground_vis(
        simulation,
        telescope,
        foreground,
        foreground_observation,
        foreground_vis_file,
        write_ms,
        foreground_ms_file,
    ):
        """
        Simulates foreground sources
        """
        print("### Simulating foreground source....")
        visibility = simulation.run_simulation(
            telescope, foreground, foreground_observation
        )
        (fg_header, fg_handle) = oskar.VisHeader.read(foreground_vis_file)
        foreground_cross_correlation = [0] * fg_header.num_blocks
        # fg_max_channel=fg_header.max_channels_per_block;
        for i in range(fg_header.num_blocks):
            fg_block = oskar.VisBlock.create_from_header(fg_header)
            fg_block.read(fg_header, fg_handle, i)
            foreground_cross_correlation[i] = fg_block.cross_correlations()
        ff_uu = fg_block.baseline_uu_metres()
        ff_vv = fg_block.baseline_vv_metres()
        ff_ww = fg_block.baseline_ww_metres()
        if write_ms:
            visibility.write_to_file(foreground_ms_file)
        return (
            visibility,
            foreground_cross_correlation,
            fg_header,
            fg_handle,
            fg_block,
            ff_uu,
            ff_vv,
            ff_ww,
        )

    @staticmethod
    def combine_vis(
        number_of_days: int, visiblity_files: list, combined_vis_filepath: str
    ):
        """
        Combine visibilities by reading visiblity_files into combined_vis_filepath
        """
        print("### Combining the visibilities for ", visiblity_files)
        out_vis = [0] * number_of_days
        uui = [0] * number_of_days
        vvi = [0] * number_of_days
        wwi = [0] * number_of_days
        time_start = [0] * number_of_days
        time_inc = [0] * number_of_days
        # for j in tqdm(range(number_of_days)):
        for j in range(number_of_days):
            (header, handle) = oskar.VisHeader.read(visiblity_files[j])
            block = oskar.VisBlock.create_from_header(header)
            for k in range(header.num_blocks):
                block.read(header, handle, k)
            out_vis[j] = block.cross_correlations()
            uui[j] = block.baseline_uu_metres()
            vvi[j] = block.baseline_uu_metres()
            wwi[j] = block.baseline_uu_metres()
            time_inc[j] = header.time_inc_sec
            time_start[j] = header.time_start_mjd_utc
        uu = np.array(uui).swapaxes(0, 1)
        uushape = uu.shape
        uu = uu.reshape(uushape[0], uushape[1] * uushape[2])
        # vv = np.array(vvi).swapaxes(0, 1);vvshape = vv.shape;vv = vv.reshape(vvshape[0], vvshape[1] * vvshape[2])
        # ww = np.array(wwi).swapaxes(0, 1);wwshape = ww.shape;ww = ww.reshape(wwshape[0], wwshape[1] * wwshape[2])
        # --------- Combining the Visibilities
        os.system("rm -rf " + combined_vis_filepath)
        ms = oskar.MeasurementSet.create(
            combined_vis_filepath,
            block.num_stations,
            block.num_channels,
            block.num_pols,
            header.freq_start_hz,
            header.freq_inc_hz,
        )
        ms.set_phase_centre(header.phase_centre_ra_deg, header.phase_centre_dec_deg)
        # Write data one block at a time.
        num_times = number_of_days
        print("### Writing combined visibilities in ", combined_vis_filepath)
        for t in range(num_times):
            # Dummy data to write.
            time_stamp = time_inc[t] * time_start[t]
            # Write coordinates and visibilities.
            start_row = t * block.num_baselines
            exposure_sec = time_inc[t]
            ms.write_coords(
                start_row,
                block.num_baselines,
                uui[t],
                vvi[t],
                wwi[t],
                exposure_sec,
                time_inc[t],
                time_stamp,
            )
            ms.write_vis(
                start_row, 0, block.num_channels, block.num_baselines, out_vis[t]
            )
