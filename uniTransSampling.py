import sys, math
import re
import time
import os

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

#https://en.wikipedia.org/wiki/Probability_integral_transform
def expCDF(L, x):
    return 1.0 - np.exp(-L*x);
def gausCDF(mu, sig, x):
    return 0.5*(1 + math.erf((x - mu)/(sig*np.sqrt(2))));

def main():
    verbosity = 1;
    binN = 100;
    rangeX = [0.0, 3.0];

    np.random.seed(1);
    expL = 1.0;
    gausMu = 1.5;
    gausSig = 0.5;
    sampleN = 10000;
#histogram
    nbins = np.linspace(rangeX[0], rangeX[1], binN);
    expPDF = np.random.exponential(1.0/expL, sampleN);
    gausPDF = np.random.normal(gausMu, gausSig, sampleN);
    expHist = np.zeros(binN);
    gausHist = np.zeros(binN);
    exp2UniHist = np.zeros(binN);
    gaus2UniHist = np.zeros(binN);
    for x in expPDF:
        if rangeX[0] < x and x < rangeX[1]:
            expHist[int(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
        uniX = expCDF(expL, x);
        if rangeX[0] < uniX and uniX < rangeX[1]:
            exp2UniHist[int(binN*(uniX-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
    for x in gausPDF: 
        if rangeX[0] < x and x < rangeX[1]:
            gausHist[int(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
        uniX = gausCDF(gausMu, gausSig, x);
        if rangeX[0] < uniX and uniX < rangeX[1]:
            gaus2UniHist[int(binN*(uniX-rangeX[0])/(rangeX[1]-rangeX[0]))] += 1;
#plots
    fig = plt.figure(figsize=(18, 14));
    gs = gridspec.GridSpec(2, 2);
    ax0 = fig.add_subplot(gs[0]);
    ax1 = fig.add_subplot(gs[1]);
    ax2 = fig.add_subplot(gs[2]);
    ax3 = fig.add_subplot(gs[3]);
    #plot 0
    ax0.plot(nbins, expHist, linewidth=2, color="blue", linestyle="steps-mid");
    ax0.set_title("Exp Distr", fontsize=24, y=1.03);
    ax0.set_xlabel("x", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0], rangeX[1]);
    ax0.set_ylim(bottom=0);
    #plot 1
    ax1.plot(nbins, gausHist, linewidth=2, color="blue", linestyle="steps-mid");
    ax1.set_title("Normal Distr", fontsize=24, y=1.03);
    ax1.set_xlabel("x", fontsize=18);
    ax1.set_ylabel("count", fontsize=18);
    ax1.set_xlim(rangeX[0], rangeX[1]);
    ax1.set_ylim(bottom=0);
    #plot 2
    ax2.plot(nbins, exp2UniHist, linewidth=2, color="blue", linestyle="steps-mid");
    ax2.set_title("Exp to Unif Distr", fontsize=24, y=1.03);
    ax2.set_xlabel("x", fontsize=18);
    ax2.set_ylabel("count", fontsize=18);
    ax2.set_xlim(rangeX[0], rangeX[1]);
    ax2.set_ylim(bottom=0);
    #plot 3
    ax3.plot(nbins, gaus2UniHist, linewidth=2, color="blue",linestyle="steps-mid");
    ax3.set_title("Normal to Unif Distr", fontsize=24, y=1.03);
    ax3.set_xlabel("x", fontsize=18);
    ax3.set_ylabel("count", fontsize=18);
    ax3.set_xlim(rangeX[0], rangeX[1]);
    ax3.set_ylim(bottom=0);
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__));
    filenameFig = exepath + "/fig/uniTransSampling.png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print(filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




