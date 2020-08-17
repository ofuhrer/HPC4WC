#!/bin/bash

# this scripts downloads test data for the physics standalone

# echo on
set -x

# get name of standalone package
cwd=`pwd`
dirname=`basename ${cwd}`

# remove preexisting data directory
test -d ./data
/bin/rm -rf data

# get data
wget --quiet "ftp://ftp.cscs.ch/in/put/abc/cosmo/fuo/physics_standalone/${dirname}/data.tar.gz"
test -f data.tar.gz || exit 1
tar -xvf data.tar.gz || exit 1
/bin/rm -f data.tar.gz 2>/dev/null

# done
