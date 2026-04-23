#!/bin/bash

set -euo pipefail

ENV_NAME="asd_agg_dl"
PHYSIOVIEW_PATH="../packages/physioview"
BASE_PATH=$(dirname $0)

cd $BASE_PATH

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f ../environment.yml

# Fetching submodules
echo "Fetching submodules..."
git submodule update --init

# Install PhysioView
echo "Installing PhysioView..."
conda run -n $ENV_NAME pip install -e $PHYSIOVIEW_PATH

echo "Setup complete!"
