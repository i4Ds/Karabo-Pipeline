# Examples

## Running a general interferometer simulation

The following example also showcases the main pipeline ingredients for a telescope simulation, similar to the line emission scripts.

```python
<example_interfe_simu.py>
```

## SRCNet

Karabo is used in the SRCNet to generate simulated test data resembling SKAO data.

The script `SRCNet_v0.1_simulation.py` (work in progress, TODO link will follow) generates simulated test data for the SRCNet v0.1 release. Data volume generated: TODO GB. Required hardware: TODO GB RAM, TODO GB storage. Approximate runtime with TODO cores: TODO h.

The notebook [SRCNet_simulation_walkthrough.ipynb](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/examples/SRCNet_simulation_walkthrough.ipynb) contains a small example based on `SRCNet_v0.1_simulation.py` that can be run on a laptop in a couple of minutes. It walks you through the whole process, from loading the survey / sky model and configuring the telescope, to configuring observation parameters and setting up and running the simulation, to creating a dirty image from the generated visibilities.

## Performing a line emission simulation, using both OSKAR and RASCIL

See the script [line_emission.py](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/simulation/line_emission.py) and the notebook [LineEmissionBackendsComparison.ipynb](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/examples/LineEmissionBackendsComparison.ipynb) for an end-to-end line emission simulation.

This simulation begins with a `SkyModel` instance, and with the definition of the desired `Observation` and `Telescope` details. Then, the `InterferometerSimulation` instance uses the requested backend (OSKAR and RASCIL are currently supported) to compute the corresponding visibilities, and the desired `DirtyImager` instance is used to convert the visibilities into dirty images. Optionally, we can include primary beam effects and correct for such effects in the final dirty images. Finally, we can mosaic different dirty images into one larger image using the `ImageMosaicker` class.

## Show telescope config

```python
<example_tel_set.py>
```

![Image](../images/telescope.png)

## Notes on the OSKAR Telescope data conventions

Karabo supports many telescope configurations, and uses the OSKAR specification for its telescope directory structure, which is explained below in more detail.

The name of the directory is given the name of the telescope followed by configuration or cycle specification e.g. for VLA configuration C will be “vla.c.tm” and for ALMA cycle 4.2, the name is “alma.cycle4.2”. The  top-level directory must contain a special file to specify the telescope centre position (position.txt), a special file to specify the position of each station (layout.txt), and a set of sub-directories one for every station. Each of these sub-directories contains one or more special files to specify the configuration of that station. For telescope with dishes, it contain just a number.

More details can be find in the OSKAR documentation and source code: https://ska-telescope.gitlab.io/sim/oskar/telescope_model/telescope_model.html

