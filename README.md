# Setup

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


# Install package

```
Add package to environment.yml
conda-lock -f environment.yml -p linux-64
conda env update --file environment.yml --prune
pip install -e . --no-build-isolation
```