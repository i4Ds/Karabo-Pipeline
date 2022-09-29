import os
import unittest
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.beam import BeamPattern
from karabo.simulation.telescope import Telescope
from karabo.test import data_path
from karabo.simulation.sky_model import SkyModel
import numpy as np
from karabo.simulation.observation import Observation
from datetime import timedelta, datetime
from karabo.imaging.imager import Imager
from astropy.io import fits
import oskar



class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make dir for result files
        if not os.path.exists("result/"):
            os.makedirs("result/")

    def test_fit_element(self):
        tel = Telescope.get_MEERKAT_Telescope()
        beam = BeamPattern(f"{data_path}/run5.cst")
        beam.fit_elements(tel, freq_hz=1.0e08, avg_frac_error=0.5)

    def test_katbeam(self):
        beampixels = BeamPattern.get_meerkat_uhfbeam(f=800, pol="I", beamextent=40)
        BeamPattern.show_kat_beam(
            beampixels, 40, 800, "I", path="./result/katbeam_beam.png"
        )

    def test_eidosbeam(self):
        npix = 500
        dia = 10
        thres = 0
        ch = 0
        B_ah = BeamPattern.get_eidos_holographic_beam(npix, ch, dia, thres, mode="AH")
        BeamPattern.show_eidos_beam(B_ah, path="./result/eidos_AH_beam.png")
        B_em = BeamPattern.get_eidos_holographic_beam(npix, ch, dia, thres, mode="EM")
        BeamPattern.show_eidos_beam(B_em, path="./result/eidos_EM_beam.png")
        BeamPattern.eidos_lineplot(
            B_ah, B_em, npix, path="./result/eidos_residual_beam.png"
        )

    def test_long_observations(self):
         number_of_days=3;hours_per_day=10
         combined_vis_filepath = './karabo/test/data/combined_vis.ms'
         #combined_vis_filepath = '/home/rohit/karabo/karabo-pipeline/karabo/test/data/combined_vis.ms'
         #-------- Iterate over days
         days = np.arange(1, number_of_days);visiblity_files=[0]*len(days);ms_files=[0]*len(days);i=0
         for day in days:
             sky = SkyModel()
             sky_data = np.array([
                  [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
                  [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
                  [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
             sky.add_point_sources(sky_data)
             telescope = Telescope.get_MEERKAT_Telescope()
             # telescope.centre_longitude = 3
             xcstfile_path='./karabo-pipeline/karabo/test/data/cst_like_beam_port_1.txt'
             ycstfile_path='./karabo-pipeline/karabo/test/data/cst_like_beam_port_2.txt'
             enable_array_beam=False
             # Remove beam if already present
             test = os.listdir(telescope.path)
             for item in test:
                 if item.endswith(".bin"):
                     os.remove(os.path.join(telescope.path, item))
             if(enable_array_beam):
                #------------ X-coordinate
                pb = BeamPattern(xcstfile_path) # Instance of the Beam class
                beam = pb.sim_beam(beam_method='Gaussian Beam') # Computing beam
                pb.save_meerkat_cst_file(beam[3]) # Saving the beam cst file
                pb.fit_elements(telescope,freq_hz=1.e9,avg_frac_error=0.8,pol='X') # Fitting the beam using cst file
                #------------ Y-coordinate
                pb=BeamPattern(ycstfile_path)
                pb.save_meerkat_cst_file(beam[4])
                pb.fit_elements(telescope, freq_hz=1.e9, avg_frac_error=0.8, pol='Y')
             print('Observing Day: '+str(day))
             #------------- Simulation Begins
             visiblity_files[i]='./karabo/test/data/beam_vis_'+str(day)+'.vis'
             ms_files[i] = visiblity_files[i].split('.vis')[0]+'.ms'
             os.system('rm -rf '+visiblity_files[i]);os.system('rm -rf '+ms_files[i])
             simulation = InterferometerSimulation(vis_path=visiblity_files[i],
                                                   channel_bandwidth_hz=2e7,
                                                   time_average_sec=1, noise_enable=False,
                                                   noise_seed="time", noise_freq="Range", noise_rms="Range",
                                                   noise_start_freq=1.e9,
                                                   noise_inc_freq=1.e6,
                                                   noise_number_freq=1,
                                                   noise_rms_start=0.1,
                                                   noise_rms_end=1,
                                                   enable_numerical_beam=enable_array_beam,enable_array_beam=enable_array_beam)
             #------------- Design Observation
             observation = Observation(mode='Tracking',phase_centre_ra_deg=20.0,
                                       start_date_and_time=datetime(2000, 1, day, 11, 00, 00, 521489),
                                       length=timedelta(hours=hours_per_day, minutes=0, seconds=0, milliseconds=0),
                                       phase_centre_dec_deg=-30.0,
                                       number_of_time_steps=1,
                                       start_frequency_hz=1.e9,
                                       frequency_increment_hz=1e6,
                                       number_of_channels=1, )
             visibility = simulation.run_simulation(telescope, sky, observation)
             visibility.write_to_file(ms_files[i])
             i=i+1
         #visibility.write_to_file("/home/rohit/karabo/karabo-pipeline/karabo/test/result/beam/beam_vis.ms")
         #---------- Reading the Visibilties --------------
         out_vis=[0]*len(days);uui=[0]*len(days);vvi=[0]*len(days);wwi=[0]*len(days); time_start=[0]*len(days); time_inc=[0]*len(days)
         for j in range(len(days)):
             (header, handle) = oskar.VisHeader.read(visiblity_files[j])
             block = oskar.VisBlock.create_from_header(header)
             for k in range(header.num_blocks):
                 block.read(header, handle, k)
             out_vis[j] = block.cross_correlations()
             uui[j]=block.baseline_uu_metres()
             vvi[j] = block.baseline_uu_metres()
             wwi[j] = block.baseline_uu_metres()
             time_inc[j]=header.time_inc_sec
             time_start[j] = header.time_start_mjd_utc
         uu=np.array(uui).swapaxes(0,1);uushape=uu.shape;uu=uu.reshape(uushape[0],uushape[1]*uushape[2])
         #vv = np.array(vvi).swapaxes(0, 1);vvshape = vv.shape;vv = vv.reshape(vvshape[0], vvshape[1] * vvshape[2])
         #ww = np.array(wwi).swapaxes(0, 1);wwshape = ww.shape;ww = ww.reshape(wwshape[0], wwshape[1] * wwshape[2])
         #--------- Combining the Visibilities
         os.system('rm -rf '+combined_vis_filepath)
         ms = oskar.MeasurementSet.create(combined_vis_filepath, block.num_stations, block.num_channels,
                                         block.num_pols, header.freq_start_hz, header.freq_inc_hz)
         ms.set_phase_centre(header.phase_centre_ra_deg, header.phase_centre_dec_deg)
         # Write data one block at a time.
         num_times=len(days)
         for t in range(num_times):
            # Dummy data to write.
            time_stamp = time_inc[t]*time_start[t]
            # Write coordinates and visibilities.
            start_row = t * block.num_baselines;exposure_sec=time_inc[t]
            ms.write_coords(start_row, block.num_baselines, uui[t], vvi[t], wwi[t],
                            exposure_sec, time_inc[t], time_stamp)
            ms.write_vis(start_row, 0, block.num_channels, block.num_baselines, out_vis[t])

         #imager = Imager(visibility, imaging_npixel=4096,imaging_cellsize=50) # imaging cellsize is over-written in the Imager based on max uv dist.
         #dirty = imager.get_dirty_image()
         #dirty.write_to_file("/home/rohit/karabo/karabo-pipeline/karabo/test/result/beam/beam_vis.fits")
         #dirty.plot(title='Flux Density (Jy)')
         #aa=fits.open('./result/beam/beam_vis.fits');bb=fits.open('/home/rohit/karabo/karabo-pipeline/karabo/test/result/beam/beam_vis_aperture.fits')
         #print(np.nanmax(aa[0].data-bb[0].data),np.nanmax(aa[0].data),np.nanmax(bb[0].data))




if __name__ == "__main__":
    unittest.main()
