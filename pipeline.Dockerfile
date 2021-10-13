FROM ubuntu:latest

ARG DEBIAN_FRONTEND="noninteractive"

COPY install.sh install.sh
RUN ./install.sh

COPY . .
ENTRYPOINT [ "python3", "pipeline.py" ]
