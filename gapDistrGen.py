import sys, math, re, time, os

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

SNUMBER = pow(10, -124);
def maxGapProb0N(x, mu, n):
    if x < SNUMBER:
        return 0;
    m = min(10, math.floor(mu/x));
    pn = 0;
    for k in range(m+1):
        pn += pow(-1, k)*special.comb(n+1, k, exact=True)*pow(1-k*x/mu, n);
    return pn;

    
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
    if incBoundary == True:
        maxNinGap += 2;
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
        if incBoundary == True:
            pickleName = "pickle/maxGapHists.pickle";
        else:
            pickleName = "pickle/maxGapHistsNoBd.pickle";
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
            if df[cond].empty == False:
                df = df.drop(df[cond].index);
            df = df.append(data, ignore_index=True);
        df.to_pickle(pickleName);
        if verbosity >= 1:
            print("The following files have been updated:");
            print("    ", pickleName);
#hist
    '''
    maxGapProbDistr = np.zeros(binN);
    for i, binX in enumerate(nbins[1:-1]):
        maxGapProbDistr[i] = maxGapProb0N(nbins[i+1], 1.0, dataPtN)\
                           - maxGapProb0N(nbins[i],   1.0, dataPtN);
    maxGapProbDistr[-1] = maxGapProb0N(rangeX[1], 1.0, dataPtN)\
                        - maxGapProb0N(nbins[i],  1.0, dataPtN);
    maxGapProbDistr = (1.0/binSize)*maxGapProbDistr;

    dataHist = np.zeros(binN);
    for x in dataPDF:
        if rangeX[0] < x and x < rangeX[1]:
            histIdx = int(np.floor(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0])));
            dataHist[histIdx] += 1;
#plots
    if verbosity >= 1:
        print("Processing the plots...");
    nbins = [entry + binSize/2.0 for entry in nbins];
    fig = plt.figure(figsize=(18, 7));
    gs = gridspec.GridSpec(1, 2);
    ax0 = fig.add_subplot(gs[0]);
    ax1 = fig.add_subplot(gs[1]); 
    #plot 0
    ax0.plot(nbins, dataHist, linewidth=2, color="blue", linestyle="steps-mid");
    ax0.axhline(y=0, xmin=0, xmax=1, color="black", linewidth=2);
    ax0.set_title("Uniform Distribution", fontsize=24, y=1.03);
    ax0.set_xlabel("x", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0]-0.1, rangeX[1]+0.1);
    #plot 1
    exepath = os.path.dirname(os.path.abspath(__file__));
    if len(maxGapHists) >= 3:
        histTop = max(maxGapHists[-3]);

    if verbosity >= 1:
        print("Creating the following files:");
    for i, hist in enumerate(maxGapHists):
        ax1.plot(nbins, hist, linewidth=2, color="blue", linestyle="steps-mid");
        ax1.axhline(y=0, xmin=0, xmax=1, color="black", linewidth=2);
        strTemp = "Max Gap Distr, N=" + str(dataPtN) + ", J=" + str(i);
        ax1.set_title(strTemp, fontsize=24, y=1.03);
        ax1.set_xlabel("max gap", fontsize=18);
        ax1.set_ylabel("count", fontsize=18);
        ax1.set_xlim(rangeX[0]-0.1, rangeX[1]+0.1);
        if i < len(maxGapHists) - 2:
            ax1.set_ylim(-math.ceil(0.1*histTop), math.ceil(1.1*histTop));
        xmin, xmax = ax1.get_xlim();
        ymin, ymax = ax1.get_ylim();
        strTemp = "sample size: " + "{:n}".format(gapSampleN);
        ax1.text(xmin+0.01*(xmax-xmin),ymax-0.04*(ymax-ymin),strTemp,fontsize=12);

        if i == 0:
            ax1.plot(nbins, maxGapProbDistr, linewidth=1.5, color="red");

        filenameFig = exepath + "/dataFig/";
        if testMode == True:
            filenameFig = filenameFig + "--TEST";
        filenameFig = filenameFig + "gapDistrN"+str(dataPtN)+"J"+str(i)+".png";
        gs.tight_layout(fig);
        plt.savefig(filenameFig);
        ax1.cla();
        if verbosity >= 1:
            print("    ", filenameFig);
        if testMode == True:
            break;
    '''
if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




