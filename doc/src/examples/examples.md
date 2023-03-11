# Examples

## Running an interferometer simulation

Running an interferometer simulation is easy.
Please look at the karabo.package documentation for specifics on the individual functions.

```python
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
import numpy as np

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

# get different predefined telescopes, like here the OSKAR example telescope.png, with a handy functions
telescope = Telescope.get_OSKAR_Example_Telescope()

# overwrite or set any of the implemented configuration values
telescope.centre_longitude = 3

simulation = InterferometerSimulation("./test_result.ms")

# create new observational settings with most settings set to default except the start frequency set to 1e6
observation = Observation(start_frequency_hz=1e6)

# run a single simulation with the provided configuration
simulation.run_simulation(telescope, sky, observation)
```
This script generates a measurement set in a file called `test_result.ms`.

## Show telescope config

```python
from karabo.simulation.telescope import Telescope

telescope = Telescope.get_OSKAR_Example_Telescope()
telescope.plot_telescope(file="example_telescope.png")
```

![Image](../images/telescope.png)
