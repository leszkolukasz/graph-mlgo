FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y \
    wget git build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -y conda-build conda-lock

WORKDIR /workspace

COPY external/llvmlite /workspace/external/llvmlite

WORKDIR /workspace/external/llvmlite

ENV GIT_DESCRIBE_TAG=0.47.0
ENV LLVMLITE_VERSION=0.47.0

RUN conda build conda-recipes/llvmdev \
    && conda build conda-recipes/llvmlite --use-local --python 3.12 \
    && conda build purge \
    && conda clean --all -y

WORKDIR /workspace
COPY environment.yml conda-lock.yml /workspace/

RUN conda-lock -f environment.yml -p linux-64 \
    && conda-lock install -n graph-mlgo conda-lock.yml \
    && conda clean --all -y

COPY . /workspace/

RUN conda run --no-capture-output -n graph-mlgo pip install --no-cache-dir -e . --no-build-isolation \
    && conda run --no-capture-output -n graph-mlgo pip cache purge

RUN echo "conda activate graph-mlgo" >> ~/.bashrc

ENV PATH="/opt/conda/envs/graph-mlgo/bin:$PATH"

CMD ["sleep", "infinity"]