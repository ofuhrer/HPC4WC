#!/bin/bash
here=$(dirname ${BASH_SOURCE[0]})

sbatch ${here}/data1.sh
sbatch ${here}/data2.sh
sbatch ${here}/data3.sh
