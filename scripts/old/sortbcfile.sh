#!/bin/bash
myfile=$1
echo $myfile
sort -n -k 2 -r $myfile > $myfile.SORTED
