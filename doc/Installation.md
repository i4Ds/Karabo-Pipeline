# Installation

To Install the package on your local system, follow the steps below.
The installation has been tested on Ubuntu 20.04 (x86_x64) and MacOS (Intel).

### Dependencies

Your system should have the following packages dependencies installed

- C and C++ compiler
- git
- git-lfs
- cmake
- python3 and pip3 (via anaconda or other python installation)
- boost-python
- casacore

The Pipeline uses the [OSKAR](https://github.com/OxfordSKA/OSKAR) and [RASCIL](https://ska-telescope.gitlab.io/external/rascil/index.html) packages.
The Installation procedure of the packages is not straight forward, therefore an installation scripts exist. See below.

### Instructions Linux (Ubuntu)

1. Install package dependencies

````shell
apt update
apt install build-essential git git-lfs cmake libboost-all-dev casacore-dev libboost-python-dev casacore-dev
````

2. Install [Anaconda or Miniconda](https://docs.anaconda.com/anaconda/install/index.html)

(Optional) 3. Create dedicated environment (environments can be python 3.7 up to python 3.9)

```shell
conda create -n pipeline_env python=3.8
conda activate pipeline_env
```

4. Install OSKAR and RASCIL

```shell
#do not run in sudo, this will install the pip packages as root (unless you want that)
#you can set the oskar installation to something in your userspace if you do not have sudo rights.
./install.sh  #oskar_install_dir=<path>
```

(Optional if 3.) 5. Activate Environment as Kernel in Jupyter Lab

```shell
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=pipeline_env
```

### Instructions MacOS (Intel)

On MacOS the installation of the dependencies can be installed most easily with [Homebrew](https://brew.sh/).
The C and C++ compilers and git are installed via the XCode Command Line Tools.

1. Install package dependencies

```shell
brew tap ska-ska/tap #tap into repository where casacore is
brew update
brew install cmake git-lfs casacore boost
```

Then either run the install.sh script and follow as on linux or proceed below.

2. Install OSKAR

There are prebuilt OSKAR [binaries](https://github.com/OxfordSKA/OSKAR/releases/download/2.7.6/OSKAR-2.7.6-Darwin.dmg) for mas os. It is the easiest to install OSKAR with those. 
Then install the python interface.:

```shell
pip install 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'
```
[Reference](https://github.com/OxfordSKA/OSKAR/blob/master/python/README.md)

3. Install Rascil

Follow the Mac OS specific [instructions](https://ska-telescope.gitlab.io/external/rascil/installation/RASCIL_macos_install.html).

### Some Remarks

Running under the new Apple Silicon is possible, however requires modification of the OSKAR Source Code or running the x86 binaries emulated.
If somebody wants to do that, please contact.
