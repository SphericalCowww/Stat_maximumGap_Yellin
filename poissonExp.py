import sys, math, re, time, os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from copy import deepcopy
from scipy import stats
from scipy import optimize
from scipy import special
from tqdm import tqdm
import pickle

import locale
locale.setlocale(locale.LC_ALL, "");
import warnings
warnings.filterwarnings("ignore");


BREAKBOUND = pow(10.0, -3);
LOWBOUND   = pow(10.0, -9);
SNUMBER    = pow(10.0, -124);


def expCDF_func(L, x):
    return 1.0 - np.exp(-L*x);
def gausCDF_func(mu, sig, x):
    return 0.5*(1 + math.erf((x - mu)/(sig*np.sqrt(2))));
def expCDF_lambda(L):
    return lambda x : expCDF_func(L, x); 
def gausCDF_lambda(mu, sig):
    return lambda x : gausCDF_func(mu, sig, x);

def poissonGetProb(n, L):
    if n == 0:
        return 1.0;
    return 1.0 - stats.poisson.cdf(n-1, L);
def poissonProb_lambda(n, mu):
    return lambda norm : poissonGetProb(n, norm*mu);
def poissonOpt_lambda(n, mu, alpha):
    return lambda norm : abs(poissonGetProb(n, norm*mu) - alpha);



def main():
    if len(sys.argv) < 2:
        print("Please input the number of signal data points.");
        print("Run stops.")
        sys.exit(0);
    verbosity = 1;
    sampleN = 3000;
    incBoundary = True;
    binN   = 1000;
    rangeX = [0.0, 5.0];
    rangeG = [1.0, 3.0];

    alpha = 0.9; 

    signalN = int(sys.argv[1]);
    #noiseN  = 0;                           #noise number
    noiseN  = 100;             #noise number
    np.random.seed(int(time.time()));
    #np.random.seed(2);

    dataN   = noiseN + signalN;
    signalMu    = 2.0;
    signalSig   = 1.0;
    noiseLambda = 1.0;
    rangeNorm = [0.0, 100.0]; binNNorm = 100;
    if dataN > 60:
        rangeNorm = [0.0, 200.0]; binNNorm = 200; 
#reading c0s, cMAXs from pickle
    pickleName = "pickle/c0cMAXHigherBounds.pickle";
    df = pd.read_pickle(pickleName);
    df = df[df["signalN"] == signalN]; 
    df = df[df["noiseN"] == noiseN];
#    df = df[df["sampleN"] == sampleN]; 
    c0s   = df["c0s"].to_numpy()[0];
    cMAXs = df["cMAXs"].to_numpy()[0];
#Poisson bound
    if verbosity >= 1:
        print("Sampling the lower bound with signalN=" + str(signalN) + ":");
    nbins = np.linspace(rangeX[0], rangeX[1], binN+1);
    nbins = nbins[:-1];
    signalCDF = gausCDF_lambda(signalMu, signalSig);    #signal shape
    mu = signalCDF(rangeG[1]) - signalCDF(rangeG[0]);
    poissons = [];
    for s in tqdm(range(sampleN)):
        noisePDF  = np.random.exponential(1.0/noiseLambda, noiseN);
        signalPDF = np.random.normal(signalMu, signalSig, signalN);
        dataPDF   = np.concatenate((noisePDF, signalPDF), axis=0);
        #getting the number of events
        dataPDF = [x for x in dataPDF if ((rangeG[0]<=x) and (x<=rangeG[1]))];
        eventN = len(dataPDF);       
 
        optFactor = 5.0;
        optRange  = [eventN/optFactor, optFactor*eventN];
        #poisson
        poissonOpt = poissonOpt_lambda(eventN, mu, alpha);
        normOpt = optimize.minimize_scalar(poissonOpt,
                    method="bounded", bounds=(optRange[0], optRange[1]),\
                    options={"xatol": LOWBOUND});
        poissonProb = poissonProb_lambda(eventN, mu);
        normUpBd = normOpt.x;
        poissons.append(normUpBd);
        if verbosity >= 2:
            print("S =", s);
            print("poi: opt norm =", normUpBd);
            print("poi: (n, mu)  =", [eventN, mu]);
            print("poi: conf (set, opt) =", [alpha, poissonProb(normUpBd)]);
    if verbosity >= 1:
        print("");
        print("c0:", c0s, "\n");
        print("cMAX:", cMAXs, "\n");
        print("Poisson:", poissons, "\n");
#pickle save
    pickleName = "c0cMAXHigherBoundsPoisson.pickle";
    try:
        df = pd.read_pickle(pickleName);
    except (OSError, IOError) as e:
        columnNames=["signalN", "noiseN", "sampleN", "incBoundary",\
                     "c0s", "cMAXs", "poissons"]; 
        df = pd.DataFrame(columns = columnNames);
    bdData = {"signalN":     signalN, \
              "noiseN":      noiseN, \
              "sampleN":     sampleN, \
              "incBoundary": incBoundary,\
              "c0s":         c0s,\
              "cMAXs":       cMAXs,\
              "poissons":    poissons};
    cond = (df["signalN"]==signalN) & (df["noiseN"]==noiseN);
    if df[cond].empty == False:
        df = df.drop(df[cond].index);
    df = df.append(bdData, ignore_index=True);
    df.to_pickle(pickleName);
    if verbosity >= 1:
        print("The following files have been updated:");
        print("    ", pickleName);
#hist
    noiseHist = np.zeros(binN);
    signalHist = np.zeros(binN);
    dataHist = np.zeros(binN);
    for x in noisePDF:
        if rangeX[0] < x and x < rangeX[1]:
            histIdx = int(np.floor(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0])));
            noiseHist[histIdx] += 1;
    for x in signalPDF: 
        if rangeX[0] < x and x < rangeX[1]:
            histIdx = int(np.floor(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0])));
            signalHist[histIdx] += 1;
    for x in dataPDF: 
        if rangeX[0] < x and x < rangeX[1]:
            histIdx = int(np.floor(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0])));
            dataHist[histIdx] += 1;
    nbinsNorm = np.linspace(rangeNorm[0], rangeNorm[1], binNNorm);
    c0Hist = np.zeros(binNNorm);
    cMAXHist = np.zeros(binNNorm);
    for norm in c0s:
        if rangeNorm[0] < norm and norm < rangeNorm[1]:
            histIdx = int(np.floor(binNNorm*(norm-rangeNorm[0])/\
                                            (rangeNorm[1]-rangeNorm[0])));
            c0Hist[histIdx] += 1;
    for norm in cMAXs:
        if rangeNorm[0] < norm and norm < rangeNorm[1]:
            histIdx = int(np.floor(binNNorm*(norm-rangeNorm[0])/\
                                            (rangeNorm[1]-rangeNorm[0])));
            cMAXHist[histIdx] += 1;
    #poisson upper bounds be discrete because eventN is discrete
    factor = 2;
    binNNorm = int(binNNorm/factor);
    nbinsNormPoi = np.linspace(rangeNorm[0], rangeNorm[1], binNNorm);
    poissonHist = np.zeros(binNNorm);
    for norm in poissons:
        if rangeNorm[0] < norm and norm < rangeNorm[1]:
            histIdx = int(np.floor(binNNorm*(norm-rangeNorm[0])/\
                                            (rangeNorm[1]-rangeNorm[0])));
            poissonHist[histIdx] += 1;
    for i, entry in enumerate(poissonHist):
        poissonHist[i] = entry/factor;
#plots
    fig = plt.figure(figsize=(18, 7));
    gs = gridspec.GridSpec(1, 2);
    ax0 = fig.add_subplot(gs[0]);
    ax1 = fig.add_subplot(gs[1]);
    #plot 0
    ax0.plot(nbins,dataHist,  linewidth=2,color="black",linestyle="steps-mid");
    ax0.plot(nbins,noiseHist, linewidth=2,color="red", linestyle="steps-mid");
    ax0.plot(nbins,signalHist,linewidth=2,color="blue",  linestyle="steps-mid");
    ax0.set_title("Generated Noise(red)&Signal(blue)", fontsize=24, y=1.03);
    ax0.set_xlabel("x", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeX[0], rangeX[1]);
    ax0.axhline(y=0, xmin=0, xmax=1, color="black", linewidth=2);
    #plot 1
    ax1.plot(nbinsNormPoi, poissonHist, color="orange", alpha=1,\
             linewidth=2, linestyle="steps-mid");
    ax1.plot(nbinsNorm, c0Hist, color="green", alpha=1,\
             linewidth=2, linestyle="steps-mid");
    ax1.plot(nbinsNorm, cMAXHist, color="purple", alpha=1,\
             linewidth=2, linestyle="steps-mid");
    ax1.set_title("Signal Upper Bound, C0(green)&CMAX(purple)&Poi(orange)", \
                  fontsize=16, y=1.03);
    ax1.set_xlabel("norm upper bound", fontsize=18);
    ax1.set_ylabel("count", fontsize=18);
    ax1.set_xlim(rangeNorm[0], rangeNorm[1]);
    ax1.axhline(y=0, xmin=0, xmax=1, color="black", linewidth=2);
    ax1.axvline(x=signalN, ymin=0, ymax=1, color="blue", linewidth=2, alpha=0.5);
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__));
    filenameFig = exepath + "/dataFig/-maxGapS";
    filenameFig = filenameFig + str(signalN) + "N" + str(noiseN) + ".png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print("    ", filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




