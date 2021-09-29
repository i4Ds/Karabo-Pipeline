# Sample Pipeline

This pipeline serves as the starting point for the SKA Digial Twin Pipeline, which is written in Python and set up in an interactive Jupyter Notebook environment. Two specific radio telescope packages are used:

- OSKAR: Responsible for the simulation of the sky and the telescope https://github.com/OxfordSKA/OSKAR
	- OSKAR telescope files telescope.tm are from https://github.com/OxfordSKA/OSKAR/releases -> Example Data
- RASCIL: Responsible for imaging https://gitlab.com/ska-telescope/external/rascil

# Setup
## Prerequisites
- Git LFS
- Virtual environment Python 3.7 (E.g. Anaconda/Miniconda)

## Installations

Details about the installations are given in the documentation of the packages. The easiest way to do this is to build from source.

1. Casacore installation: https://github.com/casacore/casacore
2. OSKAR installation: https://github.com/OxfordSKA/OSKAR & https://github.com/OxfordSKA/OSKAR/blob/master/python/README.md
3. RASCIL installation: https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html & copy the "data" directory from the pulled repository to the site-packages of the installation
