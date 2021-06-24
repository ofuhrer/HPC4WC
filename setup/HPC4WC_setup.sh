#!/bin/bash

cd ~/
if [ ! -d HPC4WC ] ; then
    echo "ERROR: Cannot find a HPC4WC repository clone in your home directory. Ask for help!"
    exit 1
fi

module load daint-gpu
module load cray-python
module load jupyter-utils

echo "Making a backup copy of HPC4WC to HPC4WC_orig"
cp -r HPC4WC HPC4WC_orig

echo "Creating virtual HPC4WC_venv Python virtual environment"
python -m venv HPC4WC_venv
source HPC4WC_venv/bin/activate
pip install -r ~/HPC4WC/setup/etc/requirements.txt

echo "Creating HPC4WC_kernel kernel for Jupyter"
cp ~/HPC4WC/setup/etc/.jupyterhub.env ~/
kernel-create -n HPC4WC_kernel
sed -i "s/if \[ \"\$SOURCE_JUPYTERHUBENV\" == true \]\; then//" ~/.local/share/jupyter/kernels/HPC4WC_kernel/launcher
sed -i "s/fi//" ~/.local/share/jupyter/kernels/HPC4WC_kernel/launcher
sed -i "s/export PYTHONPATH=''//" ~/.local/share/jupyter/kernels/HPC4WC_kernel/launcher

echo "Sucessfully finished. You must restart you JupyterHub now!"

