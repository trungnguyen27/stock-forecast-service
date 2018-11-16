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

RUN pip install --upgrade setuptools &&  python3 -m pip install -r requirements.txt

EXPOSE 80