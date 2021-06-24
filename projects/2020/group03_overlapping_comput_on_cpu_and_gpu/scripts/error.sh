#!/bin/bash

(
echo "Compiler Version Error" && \
python $(dirname ${BASH_SOURCE[0]})/error.py
) | column -t | sort -g -k 3
