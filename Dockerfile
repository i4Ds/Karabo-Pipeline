FROM jupyter/base-notebook

USER root

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libc6-dev \
    cmake \
    git \
    git-lfs \
    libboost-all-dev \
    casacore-dev \
    wget \
    qt5-default \
    g++ \
    autotools-dev \
    libicu-dev \
    libbz2-dev \
    libboost-numpy-dev  \
    libboost-python-dev  \
    software-properties-common && \
    apt-get clean

USER ${NB_UID}

RUN conda update -y conda && \
    conda install -c anaconda pip astropy && \
    conda install -c conda-forge jupyterlab matplotlib nodejs

RUN pip install dask-labextension

#install oskar

RUN mkdir oskar && \
    git clone https://github.com/OxfordSKA/OSKAR.git oskar/. && \
    mkdir oskar/build && \
    cmake -B oskar/build -S oskar/. -DCMAKE_INSTALL_PREFIX=${HOME}/oskar && \
    make -C oskar/build -j4 && \
    make -C oskar/build install

ENV OSKAR_INC_DIR "${HOME}/oskar/include"
ENV OSKAR_LIB_DIR "${HOME}/oskar/lib"
RUN pip install --user oskar/python/.

USER root

RUN wget 'https://deac-ams.dl.sourceforge.net/project/boost/boost/1.77.0/boost_1_77_0.tar.bz2' && \
    tar --bzip2 -xf boost_1_77_0.tar.bz2 -C . &&\
     cd boost_1_77_0 && \
    ./bootstrap.sh --prefix=/usr/local  --with-libraries=python && \
    ./b2 && \
    ./b2 install -d0 && \
    cd .. && \
    rm -rf boost_1_77_0 && \
    rm boost_1_77_0.tar.bz2

RUN git clone https://github.com/lofar-astron/PyBDSF.git && \
    cd PyBDSF && \
    python setup.py install && \
    cd .. && \
    rm -rf PyBDSF

#install rascil
RUN git clone https://gitlab.com/ska-telescope/external/rascil.git && \
    cd rascil && \
    pip install pip --upgrade \
    && pip install -r requirements.txt \
    && python3 setup.py install \
    && git lfs install \
    && git-lfs pull

RUN mkdir /opt/conda/lib/python3.9/site-packages/rascil-0.4.0-py3.9.egg/data
RUN cp -r rascil/data/* /opt/conda/lib/python3.9/site-packages/rascil-0.4.0-py3.9.egg/data

RUN rm -rf rascil

USER $NB_UID

RUN mkdir /home/jovyan/pipeline
COPY . /home/jovyan/pipeline/.

EXPOSE 8787
ENTRYPOINT [ "jupyter" , "lab", "--ip=0.0.0.0", "--port=8888" , "--notebook-dir=pipeline" ]
