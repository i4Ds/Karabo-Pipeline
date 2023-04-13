from __future__ import annotations

import os
import os.path
import shutil
from typing import List

import numpy as np
import oskar
from numpy.typing import NDArray

from karabo.karabo_resource import KaraboResource
from karabo.util.FileHandle import FileHandle


class Visibility(KaraboResource):
    def __init__(self, path: str = None):
        self.file = FileHandle(dir=path, suffix=".ms")

    def write_to_file(self, path: str) -> None:
        # Remove if file or folder already exists
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        shutil.copytree(self.file.path, path, dirs_exist_ok=True)

    @staticmethod
    def read_from_file(path: str) -> Visibility:
        file = FileHandle(path)
        vis = Visibility()
        vis.file = file
        return vis

    @staticmethod
    def combine_spectral_foreground_vis(
        foreground_vis_file: str,
        spectral_vis_output: List[str],
        combined_vis_filepath: str,
    ) -> None:
        """
        This function combines the visibilities of foreground and spectral lines
        Inputs: foreground visibility file, list of spectral line vis files,
        output path & name of combined vis file
        """
        print("#--- Performing visibilities combination...")
        (fg_header, fg_handle) = oskar.VisHeader.read(foreground_vis_file)
        foreground_cross_correlation: List[NDArray[np.complex64]] = list()
        fg_chan = [0] * fg_header.num_blocks
        foreground_freq = fg_header.freq_start_hz + fg_header.freq_inc_hz * np.arange(
            fg_header.num_channels_total
        )
        nvis = len(spectral_vis_output)
        out_vis: List[List[NDArray[np.complex64]]] = list()
        # fg_max_channel=fg_header.max_channels_per_block;
        for i in range(fg_header.num_blocks):
            fg_block = oskar.VisBlock.create_from_header(fg_header)
            fg_block.read(fg_header, fg_handle, i)
            fg_chan[i] = fg_block.num_channels
            foreground_cross_correlation.append(fg_block.cross_correlations())
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
            out_vis.append(list())
            for k in range(sp_header.num_blocks):
                sp_block = oskar.VisBlock.create_from_header(sp_header)
                sp_block.read(sp_header, sp_handle, k)
                out_vis[j].append(sp_block.cross_correlations())
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

    @staticmethod
    def combine_vis(
        number_of_days: int,
        visiblity_files: List[str],
        combined_vis_filepath: str,
        day_comb: bool,
    ) -> None:
        """
        Combines visibilities and writes them into into `combined_vis_filepath`.
        Args:
            number_of_days: int,
            visiblity_files: list,
            combined_vis_filepath: str,
            day_comb: bool,
        """
        print("### Combining the visibilities for ", visiblity_files)

        out_vis: List[NDArray[np.complex_]] = list()
        uui: List[NDArray[np.float_]] = list()
        vvi: List[NDArray[np.float_]] = list()
        wwi: List[NDArray[np.float_]] = list()
        time_start = list()
        time_inc = list()
        time_ave = list()
        for j in range(number_of_days):
            (header, handle) = oskar.VisHeader.read(visiblity_files[j])
            block = oskar.VisBlock.create_from_header(header)
            for k in range(header.num_blocks):
                block.read(header, handle, k)
            out_vis.append(block.cross_correlations())
            uui.append(block.baseline_uu_metres())
            vvi.append(block.baseline_vv_metres())
            wwi.append(block.baseline_ww_metres())
            time_inc.append(header.time_inc_sec)
            time_start.append(header.time_start_mjd_utc)
            time_ave.append(header.get_time_average_sec())
            print(uui[j].shape, out_vis[j].shape, number_of_days)
        # uushape = uu.shape
        # uu = uu.reshape(uushape[0], uushape[1] * uushape[2])
        # vv = np.array(vvi).swapaxes(0, 1)
        # vvshape = vv.shape
        # vv = vv.reshape(vvshape[0], vvshape[1] * vvshape[2])
        # ww = np.array(wwi).swapaxes(0, 1)
        # wwshape = ww.shape
        # ww = ww.reshape(wwshape[0], wwshape[1] * wwshape[2])
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
        deg2rad = np.pi / 180
        ms.set_phase_centre(
            header.phase_centre_ra_deg * deg2rad, header.phase_centre_dec_deg * deg2rad
        )
        # Write data one block at a time.
        print("### Writing combined visibilities in ", combined_vis_filepath)
        if day_comb:
            for j in range(number_of_days):
                num_times = out_vis[j].shape[0]
                print(num_times, out_vis[j].shape, uui[j].shape, block.num_baselines)
                for t in range(num_times):
                    # Dummy data to write.
                    time_stamp = time_inc[j] * time_start[j]
                    # Write coordinates and visibilities.
                    start_row = t * block.num_baselines
                    exposure_sec = time_ave[0]
                    # print(uui[j][t].shape,out_vis[j][t].shape,block.num_channels,block.num_baselines)
                    ms.write_coords(
                        start_row,
                        block.num_baselines,
                        uui[j][t],
                        vvi[j][t],
                        wwi[j][t],
                        exposure_sec,
                        time_ave[j],
                        time_stamp,
                    )
                    ms.write_vis(
                        start_row,
                        0,
                        block.num_channels,
                        block.num_baselines,
                        out_vis[j][t],
                    )

        if day_comb is not True:
            num_times = out_vis[j].shape[0] * number_of_days
            us = np.array(uui).shape
            outs = np.array(out_vis).shape
            uuf = np.array(uui).reshape(us[0] * us[1], us[2])
            vvf = np.array(vvi).reshape(us[0] * us[1], us[2])
            wwf = np.array(wwi).reshape(us[0] * us[1], us[2])
            out_vis_reshaped = np.array(out_vis).reshape(
                outs[0] * outs[1], outs[2], outs[3], outs[4]
            )
            for t in range(num_times):
                # Dummy data to write.
                time_stamp = time_start[0] + t * time_inc[0] / 86400.0
                # Write coordinates and visibilities.
                start_row = t * block.num_baselines
                exposure_sec = time_ave[0]
                interval_sec = time_ave[0]
                # print(time_stamp,interval_sec,exposure_sec,start_row)
                ms.write_coords(
                    start_row,
                    block.num_baselines,
                    uuf.mean(axis=0),
                    vvf.mean(axis=0),
                    wwf.mean(axis=0),
                    exposure_sec,
                    interval_sec,
                    time_stamp,
                )
                ms.write_vis(
                    start_row,
                    0,
                    block.num_channels,
                    block.num_baselines,
                    out_vis_reshaped[t],
                )
