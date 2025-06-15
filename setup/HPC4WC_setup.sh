#!/bin/bash
set -e

cd ~/
if [ ! -d HPC4WC ] ; then
    echo "ERROR: Cannot find a HPC4WC repository clone in your home directory. Ask for help!"
    exit 1
fi
if [ -d HPC4WC_venv ] ; then
    echo "ERROR: You seem to be running this command for the second time. Ask for help!"
    exit 1
fi

echo "Making a backup copy of HPC4WC to HPC4WC_orig"
cp -r HPC4WC HPC4WC_orig

echo "Creating virtual HPC4WC_venv Python virtual environment"
python -m venv HPC4WC_venv
source HPC4WC_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install setuptools wheel
python -m pip install -r ~/HPC4WC/setup/etc/requirements.txt

echo "Creating HPC4WC_kernel kernel for Jupyter"
python -m ipykernel install --user --name="HPC4WC_kernel" --display-name="HPC4WC_kernel"

echo "Sucessfully finished. You must restart your JupyterHub now!"

