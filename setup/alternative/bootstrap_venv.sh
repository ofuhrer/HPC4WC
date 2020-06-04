#!/bin/bash

MODULES=( daint-gpu cray-python/3.6.5.7 cudatoolkit )
PYTHON=python3.6
VENV=venv
FRESH_INSTALL=1

function install()
{
  source $VENV/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    MPICC=cc pip install -r requirements_daint.txt || echo "" && \
	python -m bash_kernel.install && \
	python -m gt4py.gt_src_manager install && \
	deactivate

  # On OSX only:
  # change matplotlib backend from macosx to TkAgg
  if [[ "$OSTYPE" == "darwin"* ]]
  then
    cat $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc | \
      sed -e 's/^backend.*: macosx/backend : TkAgg/g' > /tmp/.matplotlibrc && \
      cp /tmp/.matplotlibrc $VENV/lib/$PYTHON/site-packages/matplotlib/mpl-data/matplotlibrc && \
      rm /tmp/.matplotlibrc
  fi
}

for MODULE in "${MODULES[@]}"
do
  module load $MODULE || echo ""
done

if [ "$FRESH_INSTALL" -gt 0 ]
then
  echo -e "Creating new environment..."
  rm -rf $VENV
  $PYTHON -m venv $VENV
fi

install || deactivate

echo -e ""
echo -e "Command to activate environment:"
echo -e "\t\$ source $VENV/bin/activate"
echo -e ""
echo -e "Command to deactivate environment:"
echo -e "\t\$ deactivate"
echo -e ""
