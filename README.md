# GraphMLGO - Topology-aware Machine Learning Guided Compiler Optimizations Framework

This repository contains the code for the mini-paper "GraphMLGO: Topology-aware Machine Learning Guided Compiler Optimizations Framework". The paper is available in the `paper` directory.

Main contributions of this project are:
- development of a custom reinforcement learning environment designed specifically for training inlining-for-size agents.
- implementation, evaluation and comparison of two graph embedding techniques: GraphSAGE and Graph Attention Networks (GAT).

## Setup

```
conda install conda-build
conda install conda-lock

cd external/llvmlite

conda build conda-recipes/llvmdev

export GIT_DESCRIBE_TAG=0.47.0
export LLVMLITE_VERSION=0.47.0
conda build conda-recipes/llvmlite --use-local --python 3.12

cd ../..
conda-lock install -n graph-mlgo
conda activate graph-mlgo
pip install -e . --no-build-isolation
```


## Install package

```
Add package to environment.yml
conda-lock -f environment.yml -p linux-64
conda env update --file environment.yml --prune
pip install -e . --no-build-isolation
```