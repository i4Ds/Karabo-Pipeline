![Alt text](doc/src/_static/logo.png?raw=true "Karabo")
===========
[![Test Software](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yaml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/test.yaml)
[![Build Docs](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-docs.yaml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-docs.yaml)
[![Build Conda](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/conda-build.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/conda-build.yml)
[![Build Docker CLI](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-cli-docker-image.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-cli-docker-image.yml)
[![Build Docker Jupyter](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-jupyter-docker-image.yml/badge.svg)](https://github.com/i4Ds/Karabo-Pipeline/actions/workflows/build-jupyter-docker-image.yml)

[Documentation](https://i4ds.github.io/Karabo-Pipeline/) |
[Example](karabo/examples/how_to_use_karabo_example.ipynb) |
[Contributors](CONTRIBUTORS.md) |

Karabo is the name of a software system that we ambitiously call a digital twin pipeline.
It reproduces in 
a simple way the path of sky observation data through processing modules that mimic conceptually
the pipeline of the
future [Square Kilometer Array](https://www.skatelescope.org/the-ska-project/).

It should serve as a tool to learn and possibly also benchmark some of the processing steps
from a sky simulation to the science products. 

Karabo is written in Python, composed of modules that can be 
set up in an interactive Jupyter Notebook environment.

Installation
------------

The software can be installed on a normal laptop, but you will need Windows or Linux. 
On a Mac, you will need to use the [Docker](https://i4ds.github.io/Karabo-Pipeline/container.html)
version.

Please see our [documentation](https://i4ds.github.io/Karabo-Pipeline/installation_user.html) 
for the full installation instructions.


Contribute to Karabo
---------------------
We are very happy to accept contributions, either as code or even just bug reports! When writing code,
please make sure you have a quick look at our [Developer Documentation](https://i4ds.github.io/Karabo-Pipeline/development.html).
Also, feel free to file [bug reports or suggestions](https://github.com/i4Ds/Karabo-Pipeline/issues).

License
-------
© Contributors, 2023. Licensed under an [MIT License](https://github.com/i4Ds/Karabo-Pipeline/blob/main/LICENSE) license.
