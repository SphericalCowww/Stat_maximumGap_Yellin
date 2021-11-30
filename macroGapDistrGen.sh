#!/bin/bash
PYTHON_PATH=/opt/local/bin
SCRIPT_LOC=/Users/tinglin/Documents/Proj/statistics/maximumGap_Yellin

#for j in `seq 0 1 120`
#for j in `seq 101 1 120`
for j in `seq 20 1 25`
do
    time $PYTHON_PATH/python3 $SCRIPT_LOC/gapDistrGen.py $j
    time $PYTHON_PATH/python3 $SCRIPT_LOC/gapDistrDisp.py $j
done





