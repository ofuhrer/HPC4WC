#!/bin/bash

for (( i=1; i <= $1; ++i )) do
	./HPWCProject -l480 -i229 -s20 -n"$i"
done
