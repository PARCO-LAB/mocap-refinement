#!/bin/bash
i=0
max=4
for FILE in $2*; do 
    echo $FILE
    python3 filters/$1.py $1 $FILE $3 $5 &
    ((i++))
    if [ $i -gt $max ]; then
        wait
        i=0
    fi
done
wait
python3 utils/checker.py -r $4 -s $3$1/ -o $3$1.csv