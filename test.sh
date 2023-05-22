#!/bin/bash
# Usage: bash test.sh <FILTER_NAME> <input_directory> <output_directory> <reference_directory> <frequency> <temporal_window>
for FILE in $2*; do 
    echo $FILE
    python3 filters/$1.py $FILE $4 $5
done

python3 utils/checker.py -r $3 -s "${2/"input"/"output"}"$1"/" -o tmp/