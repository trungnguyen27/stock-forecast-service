FROM continuumio/miniconda3

# Identify the maintainer of an image
LABEL maintainer="quoctrunguit@gmail.com"

COPY . /app
WORKDIR /app

RUN apt-get -y update && apt-get -y install gcc
RUN conda install --verbose -c conda-forge -y fbprophet
RUN echo 'finished getting prophet'
ENV PATH /opt/conda/envs/fbprophet/bin:$PATH
RUN echo "source activate fbprophet" > ~/.bashrc
RUN python3 -m pip install -r requirements.txt