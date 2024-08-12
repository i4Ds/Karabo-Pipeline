# Examples

## Running a general interferometer simulation

The following example showcases a simple telescope simulation using the main pipeline ingredients.

```python
<example_interfe_simu.py>
```

## Imaging

The notebook [imaging.ipynb](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/examples/imaging.ipynb), shows how to use different dirty imaging and image cleaning algorithms.

## Source detection

In the example notebook [source_detection.ipynb](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/examples/source_detection.ipynb), we simulate data, perform dirty imaging, clean the image, and then run and evaluate a source detection algorithm.

## SRCNet

Karabo is used in the SRCNet to generate simulated test data resembling SKAO data.

The script [SRCNet_v0.1_simulation.py](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/examples/SRCNet_v0.1_simulation.py) generates simulated visibilities and a dirty image for the SRCNet v0.1 release.  
Data volume generated: 85 GB.  
Required hardware: 32 GB RAM, 100 GB storage.  
Approximate runtime with 16 cores: 1 day.  
Suggested command due to the runtime and due to Karabo currently logging to STDOUT: `nohup python SRCNet_v0.1_simulation.py > karabo_log.txt &`  
By default, Karabo will use /tmp as the root dir for output and the data is written to a directory created by Karabo below this root dir. If the SCRATCH env var is set, Karabo will use the directory defined there as the root dir instead. If the TMPDIR or TMP env var is set, Karabo will use the directory defined there as the root dir instead. The precedence is TMPDIR = TMP > SCRATCH > /tmp.

The notebook [SRCNet_simulation_walkthrough.ipynb](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/examples/SRCNet_simulation_walkthrough.ipynb) contains a small example based on `SRCNet_v0.1_simulation.py` that can be run on a laptop in a couple of minutes. It walks you through the whole process, from loading the survey / sky model and configuring the telescope, to configuring observation parameters and setting up and running the simulation, to creating a dirty image from the generated visibilities.

In addition, we provide utility to create SRCNet Rucio metadata for the ingestion daemon for `images` and `visibilities` data products. The example script [SRCNet_rucio_meta.py](https://github.com/i4Ds/Karabo-Pipeline/blob/main/karabo/examples/SRCNet_rucio_meta.py) provides a template, which may need some customization according to your needs (e.g. include the script into a full simulation pipeline).

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

