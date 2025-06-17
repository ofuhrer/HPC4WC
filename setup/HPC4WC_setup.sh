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

# install the kernel
python -m ipykernel install --user --name="HPC4WC_kernel" --display-name="HPC4WC_kernel"
if [ ! -d ${HOME}/.local/share/jupyter/kernels/hpc4wc_kernel ] ; then
    echo "ERROR: Problem installing the Jupyter kernel. Ask for help!"
    exit 1
fi

# install bash kernel
python -m bash_kernel.install

# add a launcher which sources the Python virtual environment first
cat > ${HOME}/.local/share/jupyter/kernels/hpc4wc_kernel/launcher <<EOF
#!/usr/bin/env bash
export PYTHONPATH=''
source \${HOME}/HPC4WC_venv/bin/activate
\${HOME}/HPC4WC_venv/bin/python -m ipykernel_launcher \$@
EOF
chmod +x ${HOME}/.local/share/jupyter/kernels/hpc4wc_kernel/launcher

# make sure Python venv is picked up in JupyterHub terminal
if ! grep -q "HPC4WC_venv/bin/activate" ${HOME}/.bashrc ; then
    echo '. ${HOME}/HPC4WC_venv/bin/activate' >> ${HOME}/.bashrc
fi

echo "Sucessfully finished. You must restart your JupyterHub now!"

