#!/bin/bash
if [ $# -lt 1 ]; then
	echo "Usage: $0 bc_output_file"
	exit 1
fi
f=$1
tail +4 $f | sort -k2nr -k1n | cut -f 1
