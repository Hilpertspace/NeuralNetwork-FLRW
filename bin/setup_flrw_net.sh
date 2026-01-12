#!/usr/bin/env bash
set -euo pipefail

INSTALL_DEV_SUITE="${1-false}"
INSTALL_NOTEBOOKS="${2-false}"

# Check whether python3.10 is installed
source ./bin/check_python_version.sh

# Install and configure pdm
source ./bin/install_pdm.sh

# Install project dependencies
source ./bin/setup_project.sh "$INSTALL_DEV_SUITE" "$INSTALL_NOTEBOOKS"
