#!/bin/bash
erank=$1
arank=$2
paste $erank $arank | awk '{print $1,$2}'
