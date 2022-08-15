#!/bin/bash
PYTHON_PATH=/opt/local/bin

#get gap distribution; indepedent from data
for j in `seq 0 1 120`
do
    time $PYTHON_PATH/python3 maxGapDistrGen.py $j
    time $PYTHON_PATH/python3 maxGapDistrDisp.py $j
done

#get confidence interval of a given signal distr and significance probability
for j in `seq 0 1 120`
do
    time $PYTHON_PATH/python3 maxGapOptIntAlpha.py $j
done

#testing one a few cases from simulated data
for j in `seq 1 1 60`
do
   time $PYTHON_PATH/python3 maxGapExp.py $j
   time $PYTHON_PATH/python3 poissonExp.py $j
done
time $PYTHON_PATH/python3 upperBoundComparison.py

