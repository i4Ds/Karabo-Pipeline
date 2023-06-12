# LEGACY-FILE, has to be checked before usage
# Create build container to not have copied filed in real container afterwards
FROM --platform=amd64 continuumio/miniconda3:4.12.0 as build
COPY environment.yaml environment.yaml
COPY requirements.txt requirements.txt

FROM --platform=amd64 continuumio/miniconda3:4.12.0
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN apt-get update && apt-get install -y curl && apt-get autoclean && rm -rf /var/lib/apt/lists/*
COPY --from=build environment.yaml environment.yaml
COPY --from=build requirements.txt requirements.txt
RUN conda update -y conda && \
    conda clean --all --yes && \
    conda install mamba -y -c conda-forge
RUN mamba env update --file environment.yaml
RUN pip install -r requirements.txt
RUN rm environment.yaml requirements.txt
RUN pip install unittest-xml-reporting
RUN mkdir /workspace
WORKDIR /workspace