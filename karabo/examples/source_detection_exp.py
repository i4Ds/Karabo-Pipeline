import os

import dask
import numpy as np
from dask.distributed import Client

from karabo.Imaging.imager import Imager
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.sourcedetection import SourceDetectionResult, SourceDetectionEvaluation

from karabo.util.dask import get_global_client, parallel_for, parallel_for_each


def one_expirement(flux):
    simulation = InterferometerSimulation(channel_bandwidth_hz=1e6,
                                          time_average_sec=10)
    observation = Observation(100e6,
                              phase_centre_ra_deg=20,
                              phase_centre_dec_deg=-30,
                              number_of_time_steps=24,
                              frequency_increment_hz=20e6,
                              number_of_channels=16)
    telescope = Telescope.get_ASKAP_Telescope()
    result = do_flux(simulation, flux, telescope, observation)
    return result


def do_flux(simulation, flux, telescope, observation):
    print(f"start with flux {flux}")
    sky = SkyModel(np.array([[20, -30, flux]]))
    visibility = simulation.run_simulation(telescope, sky, observation)
    imager = Imager(visibility,
                    imaging_npixel=4096,
                    imaging_cellsize=3.878509448876288e-05 * 2,
                    ingest_vis_nchan=16)

    dirty = imager.get_dirty_image()

    a = -0.00098910103194387 * 5
    detection = SourceDetectionResult.detect_sources_in_image(dirty, beam=(a, a, 0))

    os.mkdir(f"results_{flux}")
    dirty.write_to_file(f"results_{flux}/dirty_{flux}.fits")
    detection.write_to_file(f"results_{flux}/detection_{flux}.zip")
    evaluation = SourceDetectionEvaluation.evaluate_result_with_sky_in_pixel_space(detection, sky, 10)
    evaluation.plot(filename=f"results_{flux}/overlay.png")
    evaluation.plot_flux_histogram(filename=f"results_{flux}/flux_histogram.png")
    evaluation.plot_flux_ratio_to_ra_dec(filename=f"results_{flux}/flux_ratio_ra_dec.png")
    evaluation.plot_flux_ratio_to_distance(filename=f"results_{flux}/flux_ratio_distance.png")
    evaluation.plot_quiver_positions(filename=f"results_{flux}/quiver_positions.png")
    return detection


if __name__ == '__main__':
    client = Client(processes=False, n_workers=5)
    flux_range = np.logspace(-3, 1, 5)
    # flux_range = [1]

    # results = parallel_for_each(flux_range, one_expirement)
    for flux in flux_range:
        one_expirement(flux)
    # print(results)
