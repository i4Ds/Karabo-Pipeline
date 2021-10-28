#!/bin/bash

if [ -z "$(ls -A /home/jovyan/ska_pipeline)" ]; then
   git clone https://github.com/i4Ds/SKA.git
else
   echo "Pipeline code directory is not empty. There is code from a previous run. Skipping cloning of code."
fi

start-notebook.sh
