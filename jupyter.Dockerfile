FROM jupyter/base-notebook

ARG DEBIAN_FRONTEND="noninteractive"

USER root

RUN apt-get update #takes a while re-fetching everything
RUN apt-get install -y cmake git git-lfs libboost-all-dev casacore-dev

USER ${NB_UID}

#setup python environment
RUN pip install numpy

#install oskar
RUN mkdir oskar
RUN git clone https://github.com/OxfordSKA/OSKAR.git oskar/.
RUN mkdir oskar/build
RUN cmake -B oskar/build -S oskar/. #maybe add some more options here, via arguments?
RUN make -C oskar/build -j4
RUN make -C oskar/build install
RUN pip install oskar/python/.

#install rascil
RUN mkdir rascil/
RUN git clone https://gitlab.com/ska-telescope/external/rascil.git rascil/.
RUN pip install pip --upgrade
RUN pip install -r rascil/requirements.txt
RUN python3 rascil/setup.py install
RUN git lfs install
RUN git-lfs pull

#workaround copying the data folder into site packages
#TODO replace python version values
RUN mkdir /usr/local/lib/python3.8/dist-packages/rascil-0.3.0-py3.8.egg/data
RUN cp -r "data/"* /usr/local/lib/python3.8/dist-packages/rascil-0.3.0-py3.8.egg/data

#clean up directories
RUN rm -rf oskar
RUN rm -rf rascil

#copy pipeline-code
COPY . .
