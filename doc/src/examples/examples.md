# Examples

## Performing a line emission simulation, using both OSKAR and RASCIL

See the script `line_emission.py` and the notebook `LineEmissionBackendsComparison.ipynb` for an end-to-end line emission simulation.

This simulation begins with a `SkyModel` instance, and with the definition of the desired `Observation` and `Telescope` details. Then, the `InterferometerSimulation` instance uses the requested backend (OSKAR and RASCIL are currently supported) to compute the corresponding visibilities, and the desired `DirtyImager` instance is used to convert the visibilities into dirty images. Optionally, we can include primary beam effects and correct for such effects in the final dirty images. Finally, we can mosaic different dirty images into one larger image using the `ImageMosaicker` class.

## Running a general interferometer simulation

The following example also showcases the main pipeline ingredients for a telescope simulation, similar to the line emission scripts.

```python
from datetime import datetime

import numpy as np

from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope

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
telescope = Telescope.constructor("EXAMPLE")

# overwrite or set any of the implemented configuration values
telescope.centre_longitude = 3
simulation = InterferometerSimulation()

# create new observational settings with most settings set to default
# except the start frequency set to 1e6
observation = Observation(
    start_frequency_hz=1e6,
    start_date_and_time=datetime(2024, 3, 15, 10, 46, 0),
)

# run a single simulation with the provided configuration
simulation.run_simulation(telescope, sky, observation)
```

## Show telescope config

```python
from karabo.simulation.telescope import Telescope

telescope = Telescope.constructor("EXAMPLE")
telescope.plot_telescope(file="example_telescope.png")
```

![Image](../images/telescope.png)

## Notes on the OSKAR Telescope data conventions

Karabo supports many telescope configurations, and uses the OSKAR specification for its telescope directory structure, which is explained below in more detail.

The name of the directory is given the name of the telescope followed by configuration or cycle specification e.g. for VLA configuration C will be “vla.c.tm” and for ALMA cycle 4.2, the name is “alma.cycle4.2”. The  top-level directory must contain a special file to specify the telescope centre position (position.txt), a special file to specify the position of each station (layout.txt), and a set of sub-directories one for every station. Each of these sub-directories contains one or more special files to specify the configuration of that station. For telescope with dishes, it contain just a number.

More details can be find in the OSKAR documentation and source code: https://ska-telescope.gitlab.io/sim/oskar/telescope_model/telescope_model.html

