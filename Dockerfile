FROM python:3.6.7-jessie
COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential python3-dev gcc
RUN python3 -m pip install pip --upgrade && python3 -m pip install -r requirements.txt

EXPOSE 80