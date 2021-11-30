#!/bin/bash
PYTHON_PATH=/opt/local/bin
SCRIPT_LOC=/Users/tinglin/Documents/Proj/statistics/maximumGap_Yellin

for j in `seq 1 1 60`
do
    #time $PYTHON_PATH/python3 $SCRIPT_LOC/maxGapExp.py $j
    time $PYTHON_PATH/python3 $SCRIPT_LOC/maxGapExpReadAlpha.py $j
    time $PYTHON_PATH/python3 $SCRIPT_LOC/poissonExp.py $j
done
time $PYTHON_PATH/python3 $SCRIPT_LOC/maxGapDisp.py





