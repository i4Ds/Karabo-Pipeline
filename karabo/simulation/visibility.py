from __future__ import annotations

import os
import os.path
import shutil
from typing import List, Optional

import numpy as np
import oskar
from numpy.typing import NDArray

from karabo.karabo_resource import KaraboResource
from karabo.util.FileHandle import FileHandle


class Visibility(KaraboResource):
    def __init__(
        self,
        path: Optional[str] = None,
        ms_file_path: Optional[str] = None,
        file_name: str = "visibility",
    ) -> None:
        """
        Initializes a Visibility object.

        Parameters
        ----------
        path : Optional[str], default=None
            Specifies the path to the visibility file to be created or read.
        ms_file_path : Optional[str], default=None
            Specifies the path to the measurement set (MS) file that will be
            used to create the visibility file.
        file_name : str, default='visibility'
            Specifies the name of the visibility file to be created or read.

        Returns
        -------
        None
        """
        self.file = FileHandle(path=path, file_name=file_name, suffix=".vis")
        self.ms_file = FileHandle(path=ms_file_path, file_name=None, suffix=".MS")

    def write_to_file(self, path: str) -> None:
        # Create the directory if it does not exist
        if os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
        if os.path.isfile(self.file.path):
            shutil.copy(self.file.path, path)
        else:
            shutil.copytree(self.file.path, path, dirs_exist_ok=True)

    @staticmethod
    def read_from_file(path: str) -> Visibility:
        if Visibility.is_measurement_set(path):
            vis = Visibility(ms_file_path=path)
        else:
            vis = Visibility(path=path)
        return vis

    @staticmethod
    def is_measurement_set(path: str) -> str:
        return path.endswith(".ms") or path.endswith(".MS")

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
        visiblity_files: List[str],
        combined_ms_filepath: str,
        group_by: str = "day",
    ) -> None:
        print("Combining visibilities...")

        # Initialize lists to store data
        out_vis, uui, vvi, wwi, time_start, time_inc, time_ave = ([] for _ in range(7))

        # Loop over visibility files and read data
        for vis_file in visiblity_files:
            (header, handle) = oskar.VisHeader.read(vis_file)
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

        # Combine visibility data
        ms = oskar.MeasurementSet.create(
            combined_ms_filepath,
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

        # Write combined visibility data
        print("### Writing combined visibilities in ", combined_ms_filepath)

        num_files = len(visiblity_files)
        if group_by == "day":
            for j in range(num_files):
                num_times = out_vis[j].shape[0]
                for t in range(num_times):
                    time_stamp = time_inc[j] * time_start[j]
                    exposure_sec = time_ave[0]
                    start_row = t * block.num_baselines
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
        else:
            num_times = out_vis[0].shape[0] * num_files
            ushape = np.array(uui).shape
            outshape = np.array(out_vis).shape
            uuf = np.array(uui).reshape(ushape[0] * ushape[1], ushape[2])
            vvf = np.array(vvi).reshape(ushape[0] * ushape[1], ushape[2])
            wwf = np.array(wwi).reshape(ushape[0] * ushape[1], ushape[2])
            out_vis_reshaped = np.array(out_vis).reshape(
                outshape[0] * outshape[1], outshape[2], outshape[3], outshape[4]
            )
            for t in range(num_times):
                time_stamp = time_start[0] + t * time_inc[0] / 86400.0
                exposure_sec = time_ave[0]
                interval_sec = time_ave[0]
                start_row = t * block.num_baselines

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
