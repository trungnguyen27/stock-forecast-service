FROM python:3.6.7-jessie
COPY . /app
WORKDIR /app

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  libpng-dev \
  apt-utils \
  python-psycopg2 \
  python-dev \
  postgresql-client \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade setuptools
RUN pip install cython
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pystan
RUN pip install fbprophet
RUN pip install psycopg2
RUN pip install sqlalchemy
RUN python3 -m pip install -r requirements.txt

EXPOSE 80