FROM continuumio/miniconda3

COPY . /app
WORKDIR /app

ENV PATH /opt/conda/envs/env/bin:$PATH
RUN conda install -v gcc
RUN conda install -c conda-forge fbprophet
RUN echo "source activate fbprophet" > ~/.bashrc
RUN python3 -m pip install -r requirements.txt