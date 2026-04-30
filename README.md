# Setup

```
conda install conda-build

cd external/llvmlite

conda build conda-recipes/llvmdev

export GIT_DESCRIBE_TAG=0.47.0
export LLVMLITE_VERSION=0.47.0
conda build conda-recipes/llvmlite --use-local --python 3.12

cd ../..
conda env create -f environment.yml
conda activate graph-mlgo
pip install -e . --no-build-isolation
```