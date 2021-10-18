FROM jupyter/base-notebook

ARG DEBIAN_FRONTEND="noninteractive"
ARG PYTHON_VERSION="3.8"

USER root

RUN apt-get update #takes a while re-fetching everything
RUN apt-get install -y build-essential cmake git git-lfs casacore-dev 

USER ${NB_UID}

#RUN conda create -n pipeline_env python=3.8
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

#setup python environment
RUN pip install numpy

#install oskar
RUN mkdir oskar
RUN git clone https://github.com/OxfordSKA/OSKAR.git oskar/.
RUN mkdir oskar/build
RUN cmake -B oskar/build -S oskar/. #maybe add some more options here, via arguments?
RUN make -C oskar/build -j4
#install with eleveted priviliges
USER root
RUN make -C oskar/build install
USER ${NB_UID}
RUN pip install oskar/python/.

#USER root
#RUN wget http://sourceforge.net/projects/boost/files/boost/1.71.0/boost_1_71_0.tar.bz2 && tar xvfo boost_1_71_0.tar.bz2 && cd boost_1_71_0 && /bootstrap.sh --with-libraries=python && ./b2 toolset=gcc cxxflags=-std=gnu++0x && ./b2 install
#USER ${NB_UID}

RUN apt-get install libbost-dev-all

#install rascil
RUN mkdir rascil/
RUN git clone https://gitlab.com/ska-telescope/external/rascil.git rascil/.
RUN pip install pip --upgrade
RUN cd rascil && python3 setup.py install && cd ..
RUN cd rascil && git lfs install
RUN cd rascil && git-lfs pull

#workaround copying the data folder into site packages
#TODO replace python version values

#RUN mkdir /usr/local/lib/python3.8/dist-packages/rascil-0.3.0-py3.8.egg/data
#RUN cp -r "data/"* /usr/local/lib/python3.8/dist-packages/rascil-0.3.0-py3.8.egg/data

#clean up directories
RUN rm -rf oskar
RUN rm -rf rascil

#copy pipeline-code
COPY . .
