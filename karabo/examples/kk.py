import numpy as np

from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection import detect_sources_in_image
from karabo.Imaging.imager import Imager
from karabo.simulation.telescope import get_MEERKAT_Telescope, get_ASKAP_Telescope
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.util.dask import get_local_dask_client


def expirement():
    flux_range = np.linspace(0.5, 10, 20)

    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                          time_average_sec=10)
    observation = Observation(100e6,
                              phase_centre_ra_deg=20,
                              phase_centre_dec_deg=-30,
                              number_of_time_steps=24,
                              frequency_increment_hz=20e6,
                              number_of_channels=64)
    telescope = get_ASKAP_Telescope()

    detections = []

    for flux in flux_range:
        print(f"start with flux {flux}")
        sky = SkyModel(np.array([[20, -30, flux]]))
        visibility = simulation.run_simulation(telescope, sky, observation)
        imager = Imager(visibility, imaging_npixel=2048, imaging_cellsize=3.878509448876288e-05, ingest_vis_nchan=1)
        # deconvolved, restored, residual = imager.imaging_rascil(
        #     client=client,
        #     clean_nmajor=0,
        #     clean_algorithm='mmclean',
        #     clean_scales=[0, 6, 10, 30, 60],
        #     clean_fractional_threshold=.3,
        #     clean_threshold=.12e-3,
        #     clean_nmoment=5,
        #     clean_psf_support=640,
        #     clean_restored_output='integrated')
        # restored.plot()
        dirty = imager.get_dirty_image()
        dirty.plot()
        detection = detect_sources_in_image(dirty, beam=(0.06, 0.02, 13.3))
        detections.append(detection)


if __name__ == '__main__':
    expirement()
