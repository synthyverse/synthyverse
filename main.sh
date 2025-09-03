#!/bin/bash
set -e  # Exit on any error

GENERATOR="xgenboost"
DATASET="churn"
ENV_LOCATION="conda_env"
PYTHON_VERSION=3.10

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Initialize conda for this session
eval "$(conda shell.bash hook)"

# create conda environment
conda create --prefix $ENV_LOCATION python=$PYTHON_VERSION -y

# activate environment using full path
conda activate ./$ENV_LOCATION

# install required packages
pip install --no-cache-dir -r requirements/generators/base.txt
pip install --no-cache-dir -r requirements/generators/$GENERATOR.txt
pip install --no-cache-dir -r requirements/evaluation/eval.txt

# run script
python main.py --generator $GENERATOR --dataset $DATASET

# cleanup environment
conda deactivate
rm -rf $ENV_LOCATION