import os.path
import shutil
import oskar
from karabo.karabo_resource import KaraboResource
from karabo.util.FileHandle import FileHandle
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
    @staticmethod
    def combine_vis(number_of_days:int,visiblity_files:list, combined_vis_filepath:str):
        '''
        Combine visibilities by reading visiblity_files into combined_vis_filepath
        '''
        print("### Combining the visibilities for ", visiblity_files)
        out_vis = [0] * number_of_days;
        uui = [0] *number_of_days;
        vvi = [0] * number_of_days;
        wwi = [0] * number_of_days;
        time_start = [0] * number_of_days;
        time_inc = [0] * number_of_days
        #for j in tqdm(range(number_of_days)):
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
        uu = np.array(uui).swapaxes(0, 1);
        uushape = uu.shape;
        uu = uu.reshape(uushape[0], uushape[1] * uushape[2])
        # vv = np.array(vvi).swapaxes(0, 1);vvshape = vv.shape;vv = vv.reshape(vvshape[0], vvshape[1] * vvshape[2])
        # ww = np.array(wwi).swapaxes(0, 1);wwshape = ww.shape;ww = ww.reshape(wwshape[0], wwshape[1] * wwshape[2])
        # --------- Combining the Visibilities
        os.system('rm -rf ' + combined_vis_filepath)
        ms = oskar.MeasurementSet.create(combined_vis_filepath, block.num_stations, block.num_channels,
                                         block.num_pols, header.freq_start_hz, header.freq_inc_hz)
        ms.set_phase_centre(header.phase_centre_ra_deg, header.phase_centre_dec_deg)
        # Write data one block at a time.
        num_times = number_of_days
        print("### Writing combined visibilities in ", combined_vis_filepath)
        for t in range(num_times):
            # Dummy data to write.
            time_stamp = time_inc[t] * time_start[t]
            # Write coordinates and visibilities.
            start_row = t * block.num_baselines;
            exposure_sec = time_inc[t]
            ms.write_coords(start_row, block.num_baselines, uui[t], vvi[t], wwi[t],
                            exposure_sec, time_inc[t], time_stamp)
            ms.write_vis(start_row, 0, block.num_channels, block.num_baselines, out_vis[t])