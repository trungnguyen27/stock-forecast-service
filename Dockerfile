FROM python:3.6.7-jessie
RUN apt-get update && apt-get install -y build-essential python3-dev
RUN COPY . /app
WORKDIR /app


RUN pip install -r requirements.txt
EXPOSE 80