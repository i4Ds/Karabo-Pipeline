import os.path
import shutil
import os
from karabo.karabo_resource import KaraboResource
from karabo.util.FileHandle import FileHandle
import oskar

import numpy as np



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


    def combine_spectral_foreground_vis(foreground_vis_file,spectral_vis_output,combined_vis_filepath):
        '''
        This function combines the visibilities of foreground and spectral lines
        Inputs: foreground visibility file, list of spectral line vis files, output path & name of combined vis file
        '''
        print("#--- Performing visibilities combination...")
        (fg_header, fg_handle) = oskar.VisHeader.read(foreground_vis_file);foreground_cross_correlation=[0]*fg_header.num_blocks
        fg_chan=[0]*fg_header.num_blocks; foreground_freq = fg_header.freq_start_hz + fg_header.freq_inc_hz * np.arange(fg_header.num_channels_total)
        nvis = len(spectral_vis_output);out_vis = [0] * nvis
        #fg_max_channel=fg_header.max_channels_per_block;
        for i in range(fg_header.num_blocks):
            fg_block = oskar.VisBlock.create_from_header(fg_header)
            fg_block.read(fg_header, fg_handle, i)
            fg_chan[i]=fg_block.num_channels
            foreground_cross_correlation[i] = fg_block.cross_correlations()
        ff_uu=fg_block.baseline_uu_metres()
        ff_vv=fg_block.baseline_vv_metres()
        ff_ww=fg_block.baseline_ww_metres()
        # for j in tqdm(range(nvis)):
        for j in range(nvis):
            (sp_header, sp_handle) = oskar.VisHeader.read(spectral_vis_output[j])
            spec_freq = sp_header.freq_start_hz
            spec_idx = int(np.where(abs(foreground_freq - spec_freq) == np.min(abs(foreground_freq - spec_freq)))[0])
            print(spec_freq, spec_idx)
            out_vis[j] = [0] * sp_header.num_blocks
            for k in range(sp_header.num_blocks):
                sp_block = oskar.VisBlock.create_from_header(sp_header)
                sp_block.read(sp_header, sp_handle, k)
                out_vis[j][k] = sp_block.cross_correlations()
            block_num = int(spec_idx / fg_header.max_channels_per_block)
            chan_block_num = int(spec_idx - block_num * fg_header.max_channels_per_block)
            fcc=foreground_cross_correlation[block_num][:, chan_block_num, :, :]
            foreground_cross_correlation[block_num][:, chan_block_num, :, :]=  fcc+out_vis[j][0][0]
        # --------- Writing the Visibilities
        os.system('rm -rf ' + combined_vis_filepath)
        ms = oskar.MeasurementSet.create(combined_vis_filepath, fg_block.num_stations, fg_header.num_channels_total,
                                         fg_block.num_pols, fg_header.freq_start_hz, fg_header.freq_inc_hz)
        ms.set_phase_centre(fg_header.phase_centre_ra_deg, fg_header.phase_centre_dec_deg)
        # Write data one block at a time.
        time_inc = fg_header.time_inc_sec
        print("### Writing combined visibilities in ", combined_vis_filepath)
        start_row=0;exposure_sec=fg_header.get_time_average_sec();sc=[0]*len(foreground_cross_correlation);fg_chan=[0]*len(foreground_cross_correlation)
        time_stamp = fg_header.get_time_start_mjd_utc()
        ms.write_coords(start_row, fg_block.num_baselines, ff_uu[0], ff_vv[0], ff_ww[0],exposure_sec, time_inc, time_stamp)
        fcc_array=foreground_cross_correlation[0]
        for k in range(len(foreground_cross_correlation)-1):
            fcc_array=np.hstack((fcc_array,foreground_cross_correlation[k+1]))
        ms.write_vis(start_row, 0, fg_header.num_channels_total, fg_block.num_baselines, fcc_array)



    def simulate_foreground_vis(simulation,telescope,foreground,foreground_observation,foreground_vis_file,write_ms,foreground_ms_file):
        '''
        Simulates foreground sources
        '''
        print("### Simulating foreground source....")
        visibility = simulation.run_simulation(telescope, foreground, foreground_observation)
        (fg_header, fg_handle) = oskar.VisHeader.read(foreground_vis_file);foreground_cross_correlation=[0]*fg_header.num_blocks
        #fg_max_channel=fg_header.max_channels_per_block;
        for i in range(fg_header.num_blocks):
            fg_block = oskar.VisBlock.create_from_header(fg_header)
            fg_block.read(fg_header, fg_handle, i)
            foreground_cross_correlation[i] = fg_block.cross_correlations()
        ff_uu=fg_block.baseline_uu_metres()
        ff_vv=fg_block.baseline_vv_metres()
        ff_ww=fg_block.baseline_ww_metres()
        if(write_ms):
            visibility.write_to_file(foreground_ms_file)
        return visibility,foreground_cross_correlation,fg_header,fg_handle,fg_block,ff_uu,ff_vv,ff_ww



