#!/bin/bash
if [ $# -lt 2 ]; then
	echo "Usage $0 file_exact_bc file_appr_bc"
	exit 1
fi
fe=$1
n=$(($(sort -k2nr ${fe} | grep -m 1 -n "\t0.00$" | cut -d ':' -f 1)-1))
sort -k2nr -k1n ${fe} | head -n ${n} | cut -f 1 > ${fe}.snz
fa=$2
sort -k2nr -k1n ${fa} | head -n ${n} | cut -f 1 > ${fa}.snz
paste ${fe}.snz ${fa}.snz | ./diffrank.pl
