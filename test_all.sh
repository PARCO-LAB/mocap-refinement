#!/bin/bash

filters=($1)
errors=(missing_keypoints missing_frames missing_keypoints_with_error missing_frames_with_error)

base_path=$2

for filter in "${filters[@]}"; do
    for error in "${errors[@]}"; do
        bash test.sh $filter $base_path/test/$error/ tmp/$error/ $base_path/test/gt/ 20 $error
    done
done
