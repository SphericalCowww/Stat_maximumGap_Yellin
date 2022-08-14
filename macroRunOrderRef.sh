#!/bin/bash
PYTHON_PATH=/opt/local/bin

#get gap distribution; indepedent from data
for j in `seq 0 1 120`
do
    time $PYTHON_PATH/python3 gapDistrGen.py $j         #note: turn "testMode = False;" in the code
    time $PYTHON_PATH/python3 gapDistrDisp.py $j
done

#get confidence interval given significance probability and signal distr
for j in `seq 0 1 120`
do
    time $PYTHON_PATH/python3 maxGapMCAlpha.py $j
done

#testing one a few cases from simulated data
for j in `seq 1 1 60`
do
   #time $PYTHON_PATH/python3 maxGapExp.py $j
   time $PYTHON_PATH/python3 maxGapExpReadAlpha.py $j
   time $PYTHON_PATH/python3 poissonExp.py $j
done
time $PYTHON_PATH/python3 maxGapDisp.py

