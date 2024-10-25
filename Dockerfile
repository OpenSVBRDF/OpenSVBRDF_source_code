FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 curl libgl1 libglib2.0-0

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

COPY . /app
WORKDIR /app

RUN pip install -r /app/requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python
