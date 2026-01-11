#!/usr/bin/env bash

INSTALL_DEV_SUITE="${1-false}"
INSTALL_NOTEBOOKS="${2-false}"

echo "Install basic dependencies for FLRW-Net."
pdm install

if [ "$INSTALL_DEV_SUITE" = true ]; then
    echo "Install the DEV_SUITE for FLRW-Net."
    pdm install -G dev
fi

if [ "$INSTALL_NOTEBOOKS" = true ]; then
    echo "Install dependencies for usage of Jupyter notebooks."
    pdm install -G notebooks
fi
