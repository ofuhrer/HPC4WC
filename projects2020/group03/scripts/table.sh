#!/bin/bash

(
echo "Compiler Version Time" && \
sed -E -n -e '
	/^Running/ {
		N;N;N;N;
		/\nCrayPat/ { N }
		s/Running (\w+) (\w+)\n(CrayPat.*\n)?# ranks .*\ndata = .*\n\[\s*(\S+),\s*(\S+),\s*(\S+),\s*(\S+),\s*(\S+),\s*(\S+)\].*\n.*/\1 \2 \9/p
	}
' $1
) | column -t | sort -g -k 3
