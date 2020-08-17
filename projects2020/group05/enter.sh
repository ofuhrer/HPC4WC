docker stop hpc4wc_project 1>/dev/null 2>/dev/null
docker rm hpc4wc_project 1>/dev/null 2>/dev/null
docker run -i -t --rm \
	--mount type=bind,source=`pwd`,target=/work \
	--name=hpc4wc_project hpc4wc_project
