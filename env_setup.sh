#!/bin/bash

printf "\n*** setting up environment ***\n"

# deactivate any active conda environment or .venv
conda deactivate
# deactivate

# Define the environment name as a variable
ENV_NAME="irp_rain"
PYTHON_VERSION="3.11"

# check if the environment exists and remove it
conda env list | grep $ENV_NAME
if [ $? -eq 0 ]; then
    printf "\n*** environment $ENV_NAME exists, removing ***\n"
    conda remove --name $ENV_NAME --all -y
fi

# create conda env.
printf "\n*** creating conda env: $ENV_NAME ***\n"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# activate conda env.
conda activate $ENV_NAME

# install the package in editable mode (assuming the setup.py is in the current directory)
printf "\n*** installing package ***\n"
conda run -n $ENV_NAME pip install -e .

printf "\n *** run <conda activate $ENV_NAME> to activate the environment (unless ran .sh with <source env_setup.sh>) ***\n"