![Alt text](doc/src/_static/logo.png?raw=true "Karabo")
===========
| | |
| --- | --- |
| Testing | [![CI - Test](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yml) [![codecov](https://codecov.io/gh/i4Ds/Karabo-Pipeline/graph/badge.svg?token=WU4IC2MOXV)](https://codecov.io/gh/i4Ds/Karabo-Pipeline) |
| Package | [![Conda Latest Release](https://anaconda.org/i4ds/karabo-pipeline/badges/version.svg)](https://anaconda.org/i4ds/karabo-pipeline) [![Conda Downloads](https://anaconda.org/i4ds/karabo-pipeline/badges/downloads.svg)](https://anaconda.org/i4ds/karabo-pipeline) |
| Meta | [![License - BSD 3-Clause](https://anaconda.org/i4ds/karabo-pipeline/badges/license.svg)](https://github.com/i4Ds/Karabo-Pipeline/blob/main/LICENSE) |

[Documentation](https://i4ds.github.io/Karabo-Pipeline/) |
[Example](karabo/examples/source_detection.ipynb) |
[Contributors](CONTRIBUTORS.md)

Karabo is a radio astronomy software distribution for validation and benchmarking of radio telescopes and algorithms. It can be used to simulate the behaviour of the [Square Kilometer Array](https://www.skatelescope.org/the-ska-project/) or other supported telescopes. Our goal is to make installation and ramp-up easier for researchers and developers.

Karabo includes and relies on OSKAR, RASCIL, PyBDSF, [MIGHTEE](https://arxiv.org/abs/2211.05741), [GLEAM](https://www.mwatelescope.org/science/galactic-science/gleam/), Aratmospy, Bluebild, Eidos, Dask, Tools21cm, katbeam plus configuration of 20 well-known telescopes. Karabo can simulate instrument behavior and atmospheric effects, run imaging algorithms, and evaluate results.

<img src="https://github.com/i4Ds/Karabo-Pipeline/assets/4119188/1b5086c4-9df7-4732-a832-89fdbd8abba9" width="50%" />

You can use Karabo to build your own data processing pipelines by combinding existing libraries and your own code. Karabo is written in Python, composed of modules that can be set up in an interactive Jupyter Notebook environment.

Installation
------------

The software can be installed & used on Linux or Windows WSL.

Please see our [documentation](https://i4ds.github.io/Karabo-Pipeline/installation_user.html) 
for the full installation instructions.

We also offer [Docker](https://i4ds.github.io/Karabo-Pipeline/container.html) images.

Contribute to Karabo
---------------------
We are very happy to accept contributions, either as code or even just bug reports! When writing code,
please make sure you have a quick look at our [Developer Documentation](https://i4ds.github.io/Karabo-Pipeline/development.html).
Also, feel free to file [bug reports or suggestions](https://github.com/i4Ds/Karabo-Pipeline/issues).

License
-------
© Contributors, 2023. Licensed under an [MIT License](https://github.com/i4Ds/Karabo-Pipeline/blob/main/LICENSE) license.
