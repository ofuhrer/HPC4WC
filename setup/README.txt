1. cupy can be installed with pip:

module load daint-gpu
module load cray-python
module load cudatoolkit
pip install cupy-cuda101 --user

(or without the "--user" if you are in a venv)

I have an example notebook in /scratch/snx3000/robinson/Cupy.ipynb which you can copy to your home directory, and you should be able to execute all cells.

2. for IPyParallel we would recommend not launching the cluster from the options form. Instead our preferred method is through a set of magics.

What you need to do is load a module in your .jupyterhub.env file in your home directory:

robinson@daint102:~> cat ~/.jupyterhub.env
module load ipcmagic

I have put an example notebook /scratch/snx3000/robinson/ipc.ipynb 

3. It is not possible to turn off hyperthreading in the terminal you get with JupyterHub. Imagine it is like an "salloc -N 1"


