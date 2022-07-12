import dask
import numpy as np

from karabo.Imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import get_ASKAP_Telescope
from karabo.sourcedetection import detect_sources_in_image
from dask import delayed

from karabo.util.dask import get_global_client


def experiment():
    client = get_global_client()
    flux_range = np.logspace(-3, 1, 5)
    # flux_range = [1]

    results = []
    for flux in flux_range:
        simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                              time_average_sec=10)
        observation = Observation(100e6,
                                  phase_centre_ra_deg=20,
                                  phase_centre_dec_deg=-30,
                                  number_of_time_steps=24,
                                  frequency_increment_hz=20e6,
                                  number_of_channels=64)
        telescope = get_ASKAP_Telescope()
        result = delayed(do_flux)(simulation, flux, telescope, observation)
        results.append(result)

    results = dask.compute(*results)
    print(results)


def do_flux(simulation, flux, telescope, observation):
    print(f"start with flux {flux}")
    sky = SkyModel(np.array([[20, -30, flux]]))
    visibility = simulation.run_simulation(telescope, sky, observation)
    imager = Imager(visibility,
                    imaging_npixel=2048,
                    imaging_cellsize=3.878509448876288e-05,
                    ingest_vis_nchan=1)

    dirty = imager.get_dirty_image()
    dirty.save_as_fits(f"dirty_{flux}.fits")
    a = -0.00098910103194387 * 5
    detection = detect_sources_in_image(dirty, beam=(a, a, 0))
    detection.save_sources_to_csv(f"detection_{flux}.csv")

    return detection


if __name__ == '__main__':
    experiment()
