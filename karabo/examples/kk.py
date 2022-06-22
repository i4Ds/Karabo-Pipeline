import numpy as np

from karabo.simulation.sky_model import SkyModel
from karabo.sourcedetection import detect_sources_in_image
from karabo.Imaging.imager import Imager
from karabo.simulation.telescope import get_MEERKAT_Telescope
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.util.dask import get_local_dask_client


def expirement():
    flux_range = np.linspace(0.5, 10, 20)
    client = get_local_dask_client(5)

    observation = Observation(1e8, number_of_channels=64)
    sim = InterferometerSimulation()
    telescope = get_MEERKAT_Telescope()

    detections = []

    for flux in flux_range:
        print(f"start with flux {flux}")
        sky = SkyModel(np.array([[250, 60, flux]]))
        visibility = sim.run_simulation(telescope, sky, observation)
        imager = Imager(visibility, imaging_npixel=2048, imaging_cellsize=3.878509448876288e-05, ingest_vis_nchan=1)
        deconvolved, restored, residual = imager.imaging_rascil(
            client=client,
            clean_nmajor=0,
            clean_algorithm='mmclean',
            clean_scales=[0, 6, 10, 30, 60],
            clean_fractional_threshold=.3,
            clean_threshold=.12e-3,
            clean_nmoment=5,
            clean_psf_support=640,
            clean_restored_output='integrated')
        #detection = detect_sources_in_image(restored, beam=(0.06, 0.02, 13.3))
        #detections.append(detection)


if __name__ == '__main__':
    expirement()
