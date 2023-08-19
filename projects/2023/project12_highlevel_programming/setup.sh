#!/bin/bash
set -e

cd ~/

if [ ! -d HPC4WC ] ; then
    echo "ERROR: Cannot find a HPC4WC repository clone in your home directory."
    exit 1
fi

if [ -d HPC4WC_project12_venv ] ; then
    echo "HPC4WC_project12_venv already exists! Deleting venv and jupyter kenerl. Run script again to reinstall."
    rm -r HPC4WC_project12_venv
    rm .jupyterhub.env
    rm -r .local/share/jupyter/kernels/HPC4WC_project12_kernel
fi

module load daint-gpu
module load cray-python
module load jupyter-utils

echo "Making a backup copy of HPC4WC to HPC4WC_project12_orig"
cp -r HPC4WC HPC4WC_project12_orig

echo "Creating virtual HPC4WC_venv Python virtual environment"
python -m venv HPC4WC_project12_venv
source HPC4WC_project12_venv/bin/activate
pip install setuptools wheel
MPICC=CC pip install -r ~/HPC4WC/projects/2023/project12_highlevel_programming/requirements.txt

echo "Creating HPC4WC_project12_kernel kernel for Jupyter"
cp ~/HPC4WC/projects/2023/project12_highlevel_programming/.jupyterhub.env ~/
kernel-create -n HPC4WC_project12_kernel
sed -i "s/if \[ \"\$SOURCE_JUPYTERHUBENV\" == true \]\; then//" ~/.local/share/jupyter/kernels/HPC4WC_project12_kernel/launcher
sed -i "s/fi//" ~/.local/share/jupyter/kernels/HPC4WC_project12_kernel/launcher
sed -i "s/export PYTHONPATH=''//" ~/.local/share/jupyter/kernels/HPC4WC_project12_kernel/launcher

echo "Sucessfully finished. You must restart you JupyterHub now!"