![Alt text](doc/src/_static/logo.png?raw=true "Karabo")
===========
[![Test Software](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yml)
[![Build Docs](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-docs.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-docs.yml)
[![Build Conda](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/conda-build.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/conda-build.yml)
[![Build Docker User Image](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-user-image.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-user-image.yml)

[Documentation](https://i4ds.github.io/Karabo-Pipeline/) |
[Example](karabo/examples/source_detection.ipynb) |
[Contributors](CONTRIBUTORS.md)

Karabo is a radio astronomy software distribution for validation and benchmarking of radio telescopes and algorithms. It can be used to simulate the behaviour of the [Square Kilometer Array](https://www.skatelescope.org/the-ska-project/). Our goal is to make installation and ramp-up easier for researchers and developers.

Karabo includes and relies on OSKAR, RASCIL, PyBDSF, [MIGHTEE](https://arxiv.org/abs/2211.05741), [GLEAM](https://www.mwatelescope.org/science/galactic-science/gleam/), Aratmospy, Bluebild, Eidos, Dask, Tools21cm, katbeam plus configuration of 20 well-known telescopes. Karabo can simulate instrument behavior and atmospheric effects, run imaging algorithms, and evaluate results.

<img src="https://github.com/i4Ds/Karabo-Pipeline/assets/4119188/1b5086c4-9df7-4732-a832-89fdbd8abba9" width="50%" />

You can use Karabo to build your own data processing pipelines by combinding existing libraries and your own code. Karabo is written in Python, composed of modules that can be set up in an interactive Jupyter Notebook environment.

Installation
------------

The software can be installed on Linux, Windows or Windows WSL.

Please see our [documentation](https://i4ds.github.io/Karabo-Pipeline/installation_user.html) 
for the full installation instructions.

We also offer a [Docker](https://i4ds.github.io/Karabo-Pipeline/container.html) version.

Contribute to Karabo
---------------------
We are very happy to accept contributions, either as code or even just bug reports! When writing code,
please make sure you have a quick look at our [Developer Documentation](https://i4ds.github.io/Karabo-Pipeline/development.html).
Also, feel free to file [bug reports or suggestions](https://github.com/i4Ds/Karabo-Pipeline/issues).

License
-------
© Contributors, 2023. Licensed under an [MIT License](https://github.com/i4Ds/Karabo-Pipeline/blob/main/LICENSE) license.
