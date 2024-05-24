FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

## copy the repo
COPY ./ /workspace/latent-diffusion/
WORKDIR /workspace/latent-diffusion/

## install requirements
RUN conda env create -f environment.yaml