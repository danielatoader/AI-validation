# AI-validation

## Installation TensorRT (/w Poetry)

Install Poetry:

``` console
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt-get install pipx
pipx ensurepath
pipx install poetry
poetry config virtualenvs.in-project true
poetry env use 3.11
```

To install all packages including TensorRT
``` console
export NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=True
poetry install --no-root
```