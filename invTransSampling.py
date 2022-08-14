import sys, math, re, time, os, pathlib
from scipy.special import erfinv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from copy import deepcopy
from scipy import linalg
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings("ignore");

SNUMBER = pow(10, -124);

##################################################################################
#https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html
def expInvCDF(L, x):
    return -np.log(1 - x)/L;
def gausInvCDF(mu, sig, x):
    return erfinv(2*x - 1)*sig*np.sqrt(2) + mu;
##################################################################################
def main():
    verbosity = 1;
    binN = 100;
    rangeX = [0.0, 3.0];

    np.random.seed(2);
    sampleN = 2000;
    expL = 2.0;
    gausMu = 1.5;
    gausSig = 0.5;
#histogram
    nbins = np.linspace(rangeX[0], rangeX[1], binN);
    uniformPDF = np.random.uniform(0, 1, sampleN);
    uniformHist = np.zeros(binN);
    expHist = np.zeros(binN);
    gausHist = np.zeros(binN);
    for x in uniformPDF:
        if rangeX[0] < x and x < rangeX[1]:
            uniformHist[int(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
        expX = expInvCDF(expL, x);
        if rangeX[0] < expX and expX < rangeX[1]:
            expHist[int(binN*(expX-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
        gausX = gausInvCDF(gausMu, gausSig, x);
        if rangeX[0] < gausX and gausX < rangeX[1]:
            gausHist[int(binN*(gausX-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
#plots
    fig = plt.figure(figsize=(18, 14));
    gs = gridspec.GridSpec(2, 2);
    ax0 = fig.add_subplot(gs[0]);
    ax1 = fig.add_subplot(gs[1]);
    ax2 = fig.add_subplot(gs[2]);
    ax3 = fig.add_subplot(gs[3]);
    #plot 0
    ax0.plot(nbins, uniformHist, linewidth=2,color="blue",linestyle="steps-mid");
    ax0.set_title("Uniform Distribution", fontsize=24, y=1.03);
    ax0.set_xlabel("x", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0], rangeX[1]);
    ax0.set_ylim(bottom=0);
    #plot 1
    ax1.plot(nbins, expHist, linewidth=2,color="blue",linestyle="steps-mid");
    ax1.set_title("Exopential Distribution", fontsize=24, y=1.03);
    ax1.set_xlabel("x", fontsize=18);
    ax1.set_ylabel("count", fontsize=18);
    ax1.set_xlim(rangeX[0], rangeX[1]);
    ax1.set_ylim(bottom=0);
    #plot 2
    ax2.plot(nbins, gausHist, linewidth=2,color="blue",linestyle="steps-mid");
    ax2.set_title("Normal Distribution", fontsize=24, y=1.03);
    ax2.set_xlabel("x", fontsize=18);
    ax2.set_ylabel("count", fontsize=18);
    ax2.set_xlim(rangeX[0], rangeX[1]);
    ax2.set_ylim(bottom=0);
#save plots
    pathlib.Path("figure").mkdir(exist_ok=True);
    filenameFig = "figure/invTransSampling.png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print(filenameFig);
##################################################################################
if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




