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
SNUMBER = pow(10, -124);
def maxGapProb0N(x, mu, n):
    if x < SNUMBER: return 0;
    m = min(10, math.floor(mu/x));
    pn = 0;
    for k in range(m+1):
        pk = 1;
        for i in range(1, k+1, 1): pk = pk*(-1)*((n+1-k+i)/i)*(1-k*x/mu);
        pn += pk*pow(1-k*x/mu, n-k);
        #pn += pow(-1, k)*special.comb(n+1, k, exact=True)*pow(1-k*x/mu, n);#old
    return pn;
###################################################################################
def main():
    if len(sys.argv) < 2: 
        AssertionError("ERROR: Please input the number of data points")
    
    verbosity = 1;
    incBoundary = True;
    dataPtN = int(sys.argv[1]);
    np.random.seed(2);
#reading pickle
    if incBoundary == True: pickleName = "pickleRef/maxGapDistr.pickle";
    else:                   pickleName = "pickleRef/maxGapDistrNoBd.pickle";
    df = pd.read_pickle(pickleName);
    df = df[df["dataPtN"] == dataPtN];
    if df.empty == True:
        print("ERROR: no data available for N=" + str(dataPtN) + "."); 
        print("Run stops.")
        sys.exit(0);

    binNs   = df["binN"].values.tolist();
    rangeXs = df["range"].values.tolist();
    binN   = 0;
    rangeX = [0.0, 1.0];
    if len(np.unique(np.array(binNs))) == 1: binN = binNs[0];
    else: raise AssertionError("ERROR: the bin sizes are inconsistent");
    if len(np.unique(np.array(rangeXs))) == 2: rangeX = rangeXs[0];
    else: raise AssertionError("ERROR: the x ranges are inconsistent");
    gapSampleNs = df["gapSampleN"].values.tolist();
    inGapPtNs   = df["inGapPtN"].values.tolist();
    maxGapHists = df["PDF"].values.tolist();
#data
    nbins = np.linspace(rangeX[0], rangeX[1], binN+1)[:-1];
    binSize = 1.0*(rangeX[1]-rangeX[0])/binN;   
 
    maxGapProbDistr = np.zeros(binN);
    for i, binX in enumerate(nbins[1:-1]):
        maxGapProbDistr[i] = maxGapProb0N(nbins[i+1], 1.0, dataPtN)\
                           - maxGapProb0N(nbins[i],   1.0, dataPtN);
    maxGapProbDistr[-1] = maxGapProb0N(rangeX[1], 1.0, dataPtN)\
                        - maxGapProb0N(nbins[i],  1.0, dataPtN);
    maxGapProbDistr = (1.0/binSize)*maxGapProbDistr;

    dataPDF = np.random.uniform(rangeX[0], rangeX[1], dataPtN);    
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
    ax0.plot(nbins, dataHist, linewidth=2, color="blue", drawstyle="steps-post");
    ax0.axhline(y=0, xmin=0, xmax=1, color="black", linewidth=2);
    ax0.set_title("Uniform Distribution", fontsize=24, y=1.03);
    ax0.set_xlabel("y", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0]-0.1, rangeX[1]+0.1);
    #plot 1
    exepath = os.path.dirname(os.path.abspath(__file__));
    histTop = 0;
    if len(maxGapHists) >= 3:
        histTop = max(maxGapHists[-3]);

    if verbosity >= 1:
        print("Creating the following files:");
    for i, hist in enumerate(maxGapHists):
        gapSampleN = gapSampleNs[i];
        J = inGapPtNs[i]; 
        ax1.plot(nbins, hist, linewidth=2, color="blue", drawstyle="steps-post");
        ax1.axhline(y=0, xmin=0, xmax=1, color="black", linewidth=2);
        strTemp = "Max Gap Distr, N=" + str(dataPtN) + ", J=" + str(i);
        ax1.set_title(strTemp, fontsize=24, y=1.03);
        ax1.set_xlabel("x", fontsize=18);
        ax1.set_ylabel("count", fontsize=18);
        ax1.set_xlim(rangeX[0]-0.1, rangeX[1]+0.1);
        if i < len(maxGapHists) - 2:
            ax1.set_ylim(-math.ceil(0.1*histTop), math.ceil(1.1*histTop));
        xmin, xmax = ax1.get_xlim();
        ymin, ymax = ax1.get_ylim();
        strTemp = "sample size: " + "{:n}".format(gapSampleN);
        ax1.text(xmin+0.01*(xmax-xmin),ymax-0.04*(ymax-ymin),strTemp,fontsize=12);

        pathlib.Path("figure/maxGapDistr").mkdir(parents=True, exist_ok=True);
        filenameFig  = "figure/maxGapDistr/"
        filenameFig += "maxGapDistrN"+str(dataPtN)+"J"+str(i)+".png";
        gs.tight_layout(fig);
        plt.savefig(filenameFig);
        if i == 0:
            ax1.plot(nbins, maxGapProbDistr, linewidth=1, color="red");
            plt.savefig(filenameFig);
        if verbosity >= 1: print("    ", filenameFig);
        ax1.cla();

###################################################################################
if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




