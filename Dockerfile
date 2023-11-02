FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-pip

COPY . /app

WORKDIR /app

RUN test -f /app/generations.json && rm /app/generations.json || true

RUN pip3 install .
RUN python -m pip install git+https://github.com/huggingface/optimum-intel.git
RUN python -m pip install optimum

CMD ["python3", "main.py"]
