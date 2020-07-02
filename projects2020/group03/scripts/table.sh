#!/bin/bash

(
echo "Compiler Version Time" && \
sed -E -n -e '
	/^Running/ {
		N;N;N;N;
		s/Running (\w+) (\w+)\n# ranks .*\ndata = .*\n\[\s*(\S+),\s*(\S+),\s*(\S+),\s*(\S+),\s*(\S+),\s*(\S+)\].*\n.*/\1 \2 \8/p
	}
' $1
) | column -t | sort -g -k 3
