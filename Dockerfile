FROM python:3.6.7-jessie
RUN python --version
RUN python3 --version
RUN apt-get update && apt-get install -y build-essential python3-dev gcc
RUN python3 -m pip3 install pip3 --upgrade
COPY . /app
WORKDIR /app


RUN pip3 install -r requirements.txt
EXPOSE 80