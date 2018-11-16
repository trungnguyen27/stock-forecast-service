FROM continuumio/miniconda3

COPY . /app
WORKDIR /app

RUN conda create -n env python=3.6
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
RUN conda install -c conda-forge fbprophet
RUN conda activate fbprophet && python3 -m pip install -r requirements.txt