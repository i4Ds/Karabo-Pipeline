#!/bin/bash

# Options:
#
# Specify OSKAR Installation path:
#     -o /path/to/oskar/installation
#
#

#defaults
oskar_install_directory="/usr/local"

for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            oskar_install_dir)  oskar_install_directory=${VALUE} ;;
            *)
    esac
done

mkdir workbench
cd workbench

#install oskar
mkdir oskar
cd oskar
git clone https://github.com/OxfordSKA/OSKAR.git .
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$oskar_install_directory" # -DCMAKE_PREFIX_PATH="/usr/local/opt/qt5/" #maybe add some more options here, via arguments?
make -j4

if [ -w $oskar_install_directory ]; then
  make install
else
  #if you want to install the OSKAR in a non user folder like /usr/local/ you have to sudo during installation
  sudo make install
fi

cd ..
export OSKAR_INC_DIR="$oskar_install_directory/include"
export OSKAR_LIB_DIR="$oskar_install_directory/lib"
pip install python/.
cd ..

git clone https://github.com/lofar-astron/PyBDSF.git
cd PyBDSF
python setup.py install
cd ..
rm -rf PyBDSF

#install rascil
mkdir rascil
cd rascil
git clone https://gitlab.com/ska-telescope/external/rascil.git .
pip install pip --upgrade
pip install -r requirements.txt
python3 setup.py install
git lfs install
git-lfs pull


#workaround copying the data folder into site packages
#TODO replace python version values
CONDA_DIR=$(which anaconda)
mkdir "${CONDA_DIR%/*}/../lib/python3.8/site-packages/rascil-0.4.0-py3.8.egg/data"
cp -r "data/"* "${CONDA_DIR%/*}/../lib/python3.8/site-packages/rascil-0.4.0-py3.8.egg/data"
cd ..

#clean up directories
cd ..
rm -rf workbench
