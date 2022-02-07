FROM --platform=amd64 jupyter/base-notebook:python-3.7.6

ENV rascil_version=0.5.0
ENV oskar_version=2.7.7

USER root

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libc6-dev \
    cmake \
    git \
    git-lfs \
    casacore-dev \
    wget \
    qt5-default \
    g++ \
    autotools-dev \
    libicu-dev \
    libbz2-dev \
    gfortran \
    software-properties-common \
    libboost-all-dev && \
    apt-get clean

USER $NB_UID

RUN conda install -c anaconda pip && \
    conda install -c i4ds -c conda-forge karabo-pipeline

RUN pip install dask-labextension

##install oskar
#ENV OSKAR_INSTALL=${HOME}/oskar_
#
#RUN mkdir oskar && \
#    git clone --depth 1 --branch $oskar_version https://github.com/OxfordSKA/OSKAR.git oskar/. && \
#    mkdir oskar/build && \
#    cmake -B oskar/build -S oskar/. -DCMAKE_INSTALL_PREFIX=${OSKAR_INSTALL} && \
#    make -C oskar/build -j4 && \
#    make -C oskar/build install
#
#ENV OSKAR_INC_DIR "${OSKAR_INSTALL}/include"
#ENV OSKAR_LIB_DIR "${OSKAR_INSTALL}/lib"
#RUN pip install --user oskar/python/. && \
#    rm -rf oskar
#RUN wget 'https://deac-ams.dl.sourceforge.net/project/boost/boost/1.77.0/boost_1_77_0.tar.bz2' && \
#    tar --bzip2 -xf boost_1_77_0.tar.bz2 -C . &&\
#     cd boost_1_77_0 && \
#    ./bootstrap.sh --prefix=/usr/local  --with-libraries=python && \
#    ./b2 && \
#    ./b2 install -d0 && \
#    cd .. && \
#    rm -rf boost_1_77_0 && \
#    rm boost_1_77_0.tar.bz2

#install rascil
RUN conda install -c i4ds -c conda-forge python-casacore=3.4.0
#RUN pip install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil

RUN pip install --index-url=https://artefact.skao.int/repository/pypi-all/simple rascil
RUN mkdir rascil_data && \
    cd rascil_data && \
    curl https://ska-telescope.gitlab.io/external/rascil/rascil_data.tgz -o rascil_data.tgz && \
    tar zxf rascil_data.tgz && \
    cd data && \
    export RASCIL_DATA=`pwd`

#RUN mkdir /opt/conda/lib/python3.9/site-packages/rascil-${rascil_version}-py3.9.egg/data
#RUN cp -r rascil/data/* /opt/conda/lib/python3.9/site-packages/rascil-${rascil_version}-py3.9.egg/data
#
#RUN rm -rf rascil

USER root

RUN mkdir /home/jovyan/work/persistent/

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" && \
    fix-permissions "${HOME}/work/persistent"

ENV JUPYTER_ENABLE_LAB=yes

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

USER $NB_UID
WORKDIR $HOME