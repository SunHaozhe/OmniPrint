FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace
ADD VERSION .
COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt