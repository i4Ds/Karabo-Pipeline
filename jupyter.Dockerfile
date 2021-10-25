FROM jupyter/base-notebook

USER root

RUN apt-get clean
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
    software-properties-common

USER ${NB_UID}

RUN conda update -y conda

RUN conda install -c anaconda pip
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge matplotlib
RUN conda install -c anaconda astropy


#install oskar
RUN mkdir oskar
RUN git clone https://github.com/OxfordSKA/OSKAR.git oskar/.
RUN mkdir oskar/build
RUN cmake -B oskar/build -S oskar/. -DCMAKE_INSTALL_PREFIX=${HOME}/oskar
RUN make -C oskar/build -j4
RUN make -C oskar/build install

ENV OSKAR_INC_DIR "${HOME}/oskar/include"
ENV OSKAR_LIB_DIR "${HOME}/oskar/lib"
RUN pip install --user oskar/python/.

USER root

RUN wget 'https://deac-ams.dl.sourceforge.net/project/boost/boost/1.77.0/boost_1_77_0.tar.bz2'
RUN tar --bzip2 -xf boost_1_77_0.tar.bz2 -C .

RUN cd boost_1_77_0 && \
      ./bootstrap.sh --prefix=/usr/local  --with-libraries=python && \
      ./b2 && \
      ./b2 install -d0
RUN rm -rf boost_1_77_0

RUN git clone https://github.com/lofar-astron/PyBDSF.git
RUN cd PyBDSF && \
     python setup.py install
#USER ${NB_USER}
RUN rm -rf PyBDSF
#install rascil

RUN git clone https://gitlab.com/ska-telescope/external/rascil.git
RUN cd rascil && \
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
