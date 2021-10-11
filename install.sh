#!/bin/bash

#conda_env=$1
#conda_dir="/home//anaconda3/envs/$conda_env"

#echo $conda_dir

#if [ -d "$conda_dir" ]; 
#then
    #echo "conda has been found at $conda_dir"
#else 
    #echo "Conda has not been found at $conda_dir. You can continue the installation and the python packages to the python and pip in path" >/dev/stderr
    #while true; do
    #read -p "Do you wish to continue?" yn
    #case $yn in
        #[Yy]* ) break;;
        #[Nn]* ) exit;;
        #* ) echo "Please answer yes or no.";;
    #esac
#done
#fi

mkdir workbench
cd workbench

apt update
apt install -y cmake git git-lfs libboost-all-dev

#setup python environment
pip install numpy

#install oskar
mkdir oskar 
cd oskar 
git clone https://github.com/OxfordSKA/OSKAR.git . 
mkdir build
cd build
cmake .. #maybe add some more options here, via arguments?
make -j4
make install
cd ..
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
cd ..

#clean up directories
cd ..
rm -rf workbench
