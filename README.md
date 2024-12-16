![Alt text](doc/src/_static/logo.png?raw=true "Karabo")
===========
| | |
| --- | --- |
| Testing | [![CI - Test](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yml) [![codecov](https://codecov.io/gh/i4Ds/Karabo-Pipeline/graph/badge.svg?token=WU4IC2MOXV)](https://codecov.io/gh/i4Ds/Karabo-Pipeline) |
| Package | [![Conda Latest Release](https://anaconda.org/i4ds/karabo-pipeline/badges/version.svg)](https://anaconda.org/i4ds/karabo-pipeline) [![Conda Downloads](https://anaconda.org/i4ds/karabo-pipeline/badges/downloads.svg)](https://anaconda.org/i4ds/karabo-pipeline) |
| Meta | [![License - BSD 3-Clause](https://anaconda.org/i4ds/karabo-pipeline/badges/license.svg)](https://github.com/i4Ds/Karabo-Pipeline/blob/main/LICENSE) |

[Documentation](https://i4ds.github.io/Karabo-Pipeline/) |
[Examples](https://i4ds.github.io/Karabo-Pipeline/examples/examples.html) |
[Contributors](CONTRIBUTORS.md)

Karabo is a radio astronomy software distribution for validation and benchmarking of radio telescopes and algorithms. It can be used to simulate the behavior of the [Square Kilometer Array](https://www.skatelescope.org/the-ska-project/) or other supported telescopes. Our goal is to make installation and ramp-up easier for researchers and developers.

Karabo includes and relies on OSKAR, RASCIL, WSClean, PyBDSF, [MIGHTEE](https://arxiv.org/abs/2211.05741), [GLEAM](https://www.mwatelescope.org/science/galactic-science/gleam/), Aratmospy, Bluebild, Eidos, Dask, Tools21cm, katbeam plus configuration of 20 well-known telescopes. Karabo can simulate instrument behavior and atmospheric effects, run imaging algorithms, and evaluate results.

<img src="https://github.com/i4Ds/Karabo-Pipeline/assets/4119188/1b5086c4-9df7-4732-a832-89fdbd8abba9" width="50%" />

You can use Karabo to build your own data processing pipelines by combining existing libraries and your own code. Karabo is written in Python, composed of modules that can be set up in an interactive Jupyter Notebook environment.

Installation
------------

The software can be installed & used on Linux or Windows WSL.

Please see our [documentation](https://i4ds.github.io/Karabo-Pipeline/installation_user.html) 
for the full installation instructions.

We also offer [Docker](https://i4ds.github.io/Karabo-Pipeline/container.html) images.


Quick Look with no Installation
-------------------------------

If you are curious to see whether Karabo is for you or if you want to try it out before you install something, then this is for you: we offer a demo installation on Renkulab. This demo was created for a workshop at Swiss SKA Days in September 2024. It has been kept up to day ever since.

The demo installation can be found as [SwissSKADays-Karabo-Workshop](https://renkulab.io/projects/menkalinan56/swissskadays-karabo-workshop) once your logged in to Renkulab. The account is free. You can log in using your GitHub account, your ORCID id, or your edu-ID. You can start a server (free of cost) and start using the Karabo pipeline. However, you can also first fork the project. Changes you make will then be saved to your GitLab repository linked to your Renkulab accout.

A good starting point may be the slide deck of the workshop. You can find it in the folder `documents`. The code in the slides is available as Jupyter notebooks in the folder `notebooks`. Those help you get started.


Contribute to Karabo
---------------------
We are very happy to accept contributions, either as code or even just bug reports! When writing code,
please make sure you have a quick look at our [Developer Documentation](https://i4ds.github.io/Karabo-Pipeline/development.html).
Also, feel free to file [bug reports or suggestions](https://github.com/i4Ds/Karabo-Pipeline/issues).

License
-------
© Contributors, 2024. Licensed under an [MIT License](https://github.com/i4Ds/Karabo-Pipeline/blob/main/LICENSE) license.
