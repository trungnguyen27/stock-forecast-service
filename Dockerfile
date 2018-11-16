FROM continuumio/miniconda3

COPY . /app
WORKDIR /app

RUN apt-get -y update && apt-get -y install gcc
RUN conda install -c conda-forge fbprophet
ENV PATH /opt/conda/envs/fbprophet/bin:$PATH
RUN echo "source activate fbprophet" > ~/.bashrc
RUN python3 -m pip install -r requirements.txt