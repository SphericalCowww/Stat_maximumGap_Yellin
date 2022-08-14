import sys, math, re, time, os, pathlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from copy import deepcopy
from scipy import optimize
from scipy import special
from tqdm import tqdm
import pickle

import locale
locale.setlocale(locale.LC_ALL, "");
import warnings
warnings.filterwarnings("ignore");


###################################################################################
def main():
    testMode = False; dataPtN = 10; gapSampleN = 20000;
    if len(sys.argv) < 2:
        print("WARNING: Test Mode.");
        testMode = True;

    verbosity = 1;
    binN = 1000;
    rangeX = [0.0, 1.0];

    np.random.seed(2);
    incBoundary = True;
    if testMode == False:
        dataPtN     = int(sys.argv[1]);
        gapSampleN  = 10000000;
#data
    if verbosity >= 1:
        print("Sampling the maximum gap distributions N=" + str(dataPtN) + ":");
    nbins = np.linspace(rangeX[0], rangeX[1], binN+1);
    nbins = nbins[:-1];
    binSize = 1.0*(rangeX[1]-rangeX[0])/binN;
    maxNinGap = dataPtN - 2;
    if incBoundary == True: maxNinGap += 2;
    maxGapHists = [ np.zeros(binN) for _ in range(maxNinGap+1) ];
    for s in (tqdm(range(gapSampleN)) if verbosity>=1 else range(gapSampleN)):
        dataPDF = np.random.uniform(rangeX[0], rangeX[1], dataPtN);
        sortedPDF = np.sort(dataPDF);
        if incBoundary == True:
            sortedPDF = np.append(rangeX[0], sortedPDF);
            sortedPDF = np.append(sortedPDF, 
                                  rangeX[1] - (rangeX[1]-rangeX[0])/(10.0*binN));
        maxGapsJump = [0]*(maxNinGap+1); 
        for leftIdx, leftPoint in enumerate(sortedPDF[:-1]):
            for i, rightPoint in enumerate(sortedPDF[leftIdx+1:]):
                if (rightPoint - leftPoint) > maxGapsJump[i]:
                    maxGapsJump[i] = rightPoint - leftPoint;
        for i, maxGap in enumerate(maxGapsJump):
            histIdx = int(np.floor(binN*(maxGap-rangeX[0])/(rangeX[1]-rangeX[0])));
            maxGapHists[i][int(histIdx)]+=1;
    maxGapHists = np.array(maxGapHists)*(1.0/binSize)*(1.0/gapSampleN);
#pickle dataframe
    if testMode == False:
        pathlib.Path("pickle").mkdir(exist_ok=True);
        if incBoundary == True: pickleName = "pickle/maxGapHists.pickle";
        else:                   pickleName = "pickle/maxGapHistsNoBd.pickle";
        if verbosity >= 1:
            print("Storing the distributions in " + pickleName + "...");
        try:
            df = pd.read_pickle(pickleName);
        except (OSError, IOError) as e:
            columnNames=["gapSampleN","binN","range","dataPtN","inGapPtN","PDF"];
            df = pd.DataFrame(columns = columnNames);
        for i, hist in enumerate(maxGapHists):
            data = {"gapSampleN": gapSampleN, \
                    "binN": binN, \
                    "range": rangeX, \
                    "dataPtN": dataPtN, \
                    "inGapPtN": i, \
                    "PDF": hist};
            cond = (df["dataPtN"]==dataPtN) & (df["inGapPtN"]==i);
            if df[cond].empty == False: df = df.drop(df[cond].index);
            df = df.append(data, ignore_index=True);
        df.to_pickle(pickleName);
        if verbosity >= 1:
            print("The following files have been updated:");
            print("    ", pickleName);

###################################################################################
if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




