#!/bin/bash

while getopts o: flag
do
    case "${flag}" in
        o) oskar_install_directory=${OPTARG};;
    esac
done

mkdir workbench
cd workbench

#setup python environment
pip install numpy

#install oskar
mkdir oskar 
cd oskar 
git clone https://github.com/OxfordSKA/OSKAR.git . 
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$oskar_install_directory #maybe add some more options here, via arguments?
make -j4
make install
#sudo make install #depending on your system you might need sudo to install the OSKAR applications
cd ..
export OSKAR_INC_DIR="$oskar_install_directory/include"
export OSKAR_LIB_DIR="$oskar_install_directory/lib"
pip install python/.
cd ..

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
mkdir $HOME/anaconda3/lib/python3.8/site-packages/rascil-0.3.0-py3.8.egg/data
cp -r "data/"* $HOME/anaconda3/lib/python3.8/site-packages/rascil-0.3.0-py3.8.egg/data
cd ..

#clean up directories
cd ..
rm -rf workbench
