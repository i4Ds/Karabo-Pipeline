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

Karabo is a starting point for the [Square Kilometer Array](https://www.skatelescope.org/the-ska-project/) Digital Twin Pipeline, which is written in Python and set up in an interactive Jupyter Notebook environment.

Two specific radio telescope packages are used:

- OSKAR: Responsible for the simulation of the sky and the telescope https://github.com/OxfordSKA/OSKAR
	- OSKAR telescope files telescope.tm are from https://github.com/OxfordSKA/OSKAR/releases -> Example Data
- RASCIL: Responsible for imaging https://gitlab.com/ska-telescope/external/rascil

Requirements
------------

- Linux or Windows with WSL. 
- Conda

Installation
------------

```shell
conda create -n karabo-env python=3.9
conda activate karabo-env
conda install -c i4ds -c conda-forge karabo-pipeline
```

For further details check our documentation:
https://i4ds.github.io/Karabo-Pipeline/installation_user.html

License
-------
© Contributors, 2022. Licensed under an [MIT License](https://github.com/i4Ds/Karabo-Pipeline/blob/main/LICENSE) license.

Contribute to Karabo
---------------------
Please have a look at our [issues](https://github.com/i4Ds/Karabo-Pipeline/issues).

