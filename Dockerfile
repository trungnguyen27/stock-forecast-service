FROM quoctrunguit/stocker
# Identify the maintainer of an image
LABEL maintainer="quoctrunguit@gmail.com"

COPY . /app
WORKDIR /app

RUN apt-get -y update && apt-get -y install gcc && conda update -y conda && conda install -c conda-forge -y fbprophet
ENV PATH /opt/conda/envs/fbprophet/bin:$PATH
RUN echo "source activate fbprophet" > ~/.bashrc && python3 -m pip install -r requirements.txt

# Run server
EXPOSE 8000
CMD gunicorn stocker_app.application:app