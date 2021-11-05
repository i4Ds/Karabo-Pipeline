FROM --platform=amd64 jupyter/base-notebook

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
    software-properties-common \
    nano && \
    apt-get clean

USER $NB_UID

RUN conda update -y conda && \ 
    conda install -c anaconda pip astropy && \
    conda install -c conda-forge matplotlib nodejs

RUN pip install dask-labextension   

#install oskar
ENV OSKAR_INSTALL=${HOME}/oskar_

RUN mkdir oskar && \
    git clone https://github.com/OxfordSKA/OSKAR.git oskar/. && \
    mkdir oskar/build && \
    cmake -B oskar/build -S oskar/. -DCMAKE_INSTALL_PREFIX=${OSKAR_INSTALL} && \
    make -C oskar/build -j4 && \
    make -C oskar/build install

ENV OSKAR_INC_DIR "${OSKAR_INSTALL}/include"
ENV OSKAR_LIB_DIR "${OSKAR_INSTALL}/lib"
RUN pip install --user oskar/python/. && \
    rm -rf oskar

USER root

RUN wget 'https://deac-ams.dl.sourceforge.net/project/boost/boost/1.77.0/boost_1_77_0.tar.bz2' && \
     tar --bzip2 -xf boost_1_77_0.tar.bz2 -C . &&\
      cd boost_1_77_0 && \
     ./bootstrap.sh --with-libraries=python && \
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

RUN mkdir /home/jovyan/work/persistent/
COPY docker-start.sh docker-start.sh
RUN  chmod +x docker-start.sh

RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" && \
    fix-permissions "${HOME}/work/persistent"

ENV JUPYTER_ENABLE_LAB=yes

ENTRYPOINT ["tini", "-g", "--"]
CMD ["./docker-start.sh"]

# USER $NB_UID
WORKDIR $HOME
