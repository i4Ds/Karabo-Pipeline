import numpy as np

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.util.file_handle import FileHandle

# create a simple sky model with three point sources
sky = SkyModel()
sky_data = np.array(
    [
        [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0, 0, 0],
        [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50, 45],
        [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10],
    ]
)
sky.add_point_sources(sky_data)

# get different predefined telescopes,
# like here the OSKAR example telescope.png, with a handy functions
telescope = Telescope.get_OSKAR_Example_Telescope()

# overwrite or set any of the implemented configuration values
telescope.centre_longitude = 3

# Create path
fh = FileHandle("./test_result.ms")
simulation = InterferometerSimulation(fh.path)

# create new observational settings with most settings set to default
# except the start frequency set to 1e6
observation = Observation(start_frequency_hz=1e6)

# run a single simulation with the provided configuration
simulation.run_simulation(telescope, sky, observation)

# clean up
fh.clean_up()
