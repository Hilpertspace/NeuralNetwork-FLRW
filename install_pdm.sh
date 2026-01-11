#!/usr/bin/env bash

echo "Install and upgrade pip."
python -m pip install --upgrade pip

echo "Install and upgrade pdm."
python -m pip install --upgrade pdm

echo "Configure pdm to use the found installation of python3.10"
pdm use -f "$PYTHON_CMD"

pdm info
