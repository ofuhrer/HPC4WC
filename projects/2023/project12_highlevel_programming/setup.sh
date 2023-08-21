#!/bin/bash
set -e

cd ~/

if [ ! -d HPC4WC_project12 ] ; then
    echo "ERROR: Cannot find a HPC4WC_project12 repository clone in your home directory."
    exit 1
fi

if [ -d HPC4WC_project12_venv ] ; then
    echo "HPC4WC_project12_venv already exists! Deleting venv. Run script again to reinstall."
    rm -r HPC4WC_project12_venv
fi

module load daint-gpu
module load cray-python
module load jupyter-utils

echo "Making a backup copy of HPC4WC to HPC4WC_project12_orig"
cp -r HPC4WC_project12 HPC4WC_project12_orig

echo "Creating virtual HPC4WC_venv Python virtual environment"
python -m venv HPC4WC_project12_venv
source HPC4WC_project12_venv/bin/activate
pip install setuptools wheel
MPICC=CC pip install -r ~/HPC4WC_project12/projects/2023/project12_highlevel_programming/requirements.txt

if [ -d .local/share/jupyter/kernels/HPC4WC_kernel ] ; then
    echo "HPC4WC_kernel already exists. No need to reinstall."
    echo "Sucessfully finished. You must restart you JupyterHub now!"
    exit 0
fi

echo "Creating HPC4WC_kernel kernel for Jupyter"
cp ~/HPC4WC/setup/etc/.jupyterhub.env ~/
kernel-create -n HPC4WC_kernel
sed -i "s/if \[ \"\$SOURCE_JUPYTERHUBENV\" == true \]\; then//" ~/.local/share/jupyter/kernels/HPC4WC_kernel/launcher
sed -i "s/fi//" ~/.local/share/jupyter/kernels/HPC4WC_kernel/launcher
sed -i "s/export PYTHONPATH=''//" ~/.local/share/jupyter/kernels/HPC4WC_kernel/launcher

echo "Sucessfully finished. You must restart you JupyterHub now!"