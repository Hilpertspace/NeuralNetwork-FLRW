#!/usr/bin/env bash
# set -euxo pipefail

# Check whether python3.10 is installed
source ./check_python_version.sh
echo "$PYTHON_CMD"

# Install and configure pdm
source ./install_pdm.sh

# Install project dependencies
source ./setup_project.sh
