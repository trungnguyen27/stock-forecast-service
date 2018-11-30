FROM quoctrunguit/prophet
# Identify the maintainer of an image
LABEL maintainer="quoctrunguit@gmail.com"

COPY . /app
WORKDIR /app