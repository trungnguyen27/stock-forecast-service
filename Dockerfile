FROM continuumio/miniconda3

# Identify the maintainer of an image
LABEL maintainer="quoctrunguit@gmail.com"

COPY . /app
WORKDIR /app

RUN apt-get -y update && apt-get -y install gcc
RUN conda update -y conda && conda create --name fbprophet 
ENV PATH /opt/conda/envs/fbprophet/bin:$PATH
RUN echo "source activate fbprophet" > ~/.bashrc && python3 -m pip install -r requirements.txt

# Run server
EXPOSE 5000
CMD gunicorn -b 0.0.0.0:8000 --access-logfile - "application"