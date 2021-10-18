FROM ubuntu:latest

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && \
    apt-get install -y cmake git git-lfs python3 python3-pip libboost-all-dev casacore-dev

COPY install.sh install.sh
RUN ./install.sh

COPY . .
ENTRYPOINT [ "python3", "pipeline.py" ]
