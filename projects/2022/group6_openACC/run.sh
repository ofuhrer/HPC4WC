#!bin/bash

# load modules
module load PrgEnv-cray
module load daint-gpu
module load craype-accel-nvidia60

# read version of the code
vers=$(find stencil2d*.F90 | xargs -i sh -c 'var="{}"; var="${var##*-}"; echo "${var%%.*}"')
echo "Which version? (options: $(echo $vers | xargs))"
read ver

# extract the best version of the GPU code
ver_best=$(echo $vers | tr " " "\n" | awk '/acc/{print}' | xargs | awk '{print $NF}')

# compile the code
make VERSION=$ver

# read domain size
size_std=128
echo "Value of nx=ny? (standard: ${size_std})"
read size
if [ ${#size} == 2 ]; then
	size=00${size}
else
	if [ ${#size} == 3 ]; then
		size=0${size}
	fi
fi

# run the code
out_file="result_${ver}_${size}.txt"
srun -n 1 -A class03 -C gpu ./stencil2d-${ver}.x --nx ${size} --ny ${size} --nz 64 --num_iter 1024 > ${out_file}

# delete clutter files
make clean

# add version and size to .dat files
for file in *.dat; do
	mv "$file" "${file/.dat/_${ver}_${size}.dat}"
done

# move .dat files and .txt files to the data directory
find *.dat *.txt -exec sh -c 'var={}; mv "$var" "data/$var"' \;



