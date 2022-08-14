import sys, math, re, time, os, pathlib

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

###################################################################################
BREAKBOUND = pow(10.0, -3);
LOWBOUND   = pow(10.0, -9);
SNUMBER    = pow(10.0, -124);
def maxGapProb0N(x, mu, n):
    if x < SNUMBER:
        return 0;
    m = min(10, math.floor(mu/x));
    pn = 0;
    for k in range(m+1):
        pn += pow(-1, k)*special.comb(n+1, k, exact=True)*pow(1-k*x/mu, n);
    return pn;
def mGapAnaProb0K(x, mu, k):
    if k == 0:
        return 1.0;
    pk = 1.0;
    b = k*x - mu; 
    for i in range(1, k, 1):
        pk = pk*(b*math.exp(-x)/i);     #avoiding overflow
        if abs(pk) < LOWBOUND:
            return 0.0;    
    pk = pk*math.exp(-x)*(b - k);
    return pk;
def mGapAnaProb0(x, mu):
    c0 = 0;
    '''
    if mu < 1.0:
        if x < 1.0:
            c0 = 0.0;
        else:
            c0 = 1.0;
    elif mu < x:
        c0 = 1;
    elif ((x < 0.03) and ( mu <  60.0)) or\
         ((x < 1.8)  and ( mu >= 60.0)) or\
         ((x < 4.0)  and ( mu >  700.0)):
        C0 = 0;
    else:
    '''
    for k in range(math.floor(1.0*mu/x) + 1):
        c0 += mGapAnaProb0K(x, mu, k);
    return c0;
def mGapAnaProb0_lambda(x, mu):
    return lambda norm : mGapAnaProb0(x*norm, mu*norm);
def mGapAnaOpt_lambda(x, mu, alpha):
    return lambda norm : abs(mGapAnaProb0(x*norm, mu*norm) - alpha);

def expCDF_func(L, x):
    return 1.0 - np.exp(-L*x);
def gausCDF_func(mu, sig, x):
    return 0.5*(1 + math.erf((x - mu)/(sig*np.sqrt(2))));
def expCDF_lambda(L):
    return lambda x : expCDF_func(L, x); 
def gausCDF_lambda(mu, sig):
    return lambda x : gausCDF_func(mu, sig, x);

def mGapMCGetCDF(binN, rangeX, pdf, x):
    nbins = np.linspace(rangeX[0], rangeX[1], binN);
    highIdx = 0;
    while nbins[highIdx] < x:
        highIdx += 1;
    lowIdx = highIdx - 1;
    cdf_x = sum(pdf[0:lowIdx])*(1.0*nbins[1]-nbins[0]);
    #gapDistrGen uses int; histogram binning on the lower bound
    cdf_x += pdf[lowIdx]*(x - nbins[lowIdx]);
    return cdf_x;
def mGapMCGetProb(CDFs_x, Ns, L): #note: both gap&poisson from lower
    c = 0;
    cTemp = 1;
    recordTemp = False;
    for n in Ns:
        if n != 0:
            cTemp *= 1.0*L/n;
        if CDFs_x[n] != 0:
            c += cTemp*math.exp(-L)*CDFs_x[n];
        if abs(1.0 - CDFs_x[n]) < BREAKBOUND:
            break;
    c += 1.0 - stats.poisson.cdf(n, L);
    return c;
def mGapMCProb_lambda(CDFs_x, Ns, mu):
    return lambda norm : mGapMCGetProb(CDFs_x, Ns, norm*mu);
#note: both gap and poisson are CDF; summation starts from the lowest value
#      Thence, the norm is the upper bound
def mGapMCOpt_lambda(CDFs_x, Ns, mu, alpha):
    return lambda norm : abs(mGapMCGetProb(CDFs_x, Ns, norm*mu) - alpha);

###################################################################################
def main():
    if len(sys.argv) < 2:
        print("Please input the number of signal data points.");
        print("Run stops.")
        sys.exit(0);
    verbosity = 1;
    sampleN = 3000;
    incBoundary = True;
    binN   = 1000;
    rangeX = [-3.0, 3.0];
    rangeG = [-1.0, 1.0];

    alphas = [0.90, 0.93, 0.95, 0.96, 0.97, 0.98];
    signalN = int(sys.argv[1]);
    np.random.seed(signalN);
    #np.random.seed(2);

    signalMu    = 0.0;
    signalSig   = 1.0;
    rangeNorm = [0.0, 100.0]; binNNorm = 100;
    if signalN > 60: rangeNorm = [0.0, 200.0]; binNNorm = 200; 
#dataframe from pickle
    if incBoundary == True: pickleName = "pickleRef/maxGapDistr.pickle";
    else:                   pickleName = "pickleRef/maxGapDistrNoBd.pickle";
    df = pd.read_pickle(pickleName);
    NN = df[df["inGapPtN"] == 0]["dataPtN"].size;
    mGapPDFss = [0]*NN;
    mGapNss   = [0]*NN;
    mGapbinN   = 0;
    mGapRangeX = [0.0, 1.0];
    for J in range(NN):
        cJdf = df[df["inGapPtN"] == J].sort_values("dataPtN");
        mGapbinNs   = cJdf["binN"].to_numpy();
        mGapRangeXs = cJdf["range"].to_numpy();
        if len(np.unique(mGapbinNs)) == 1:
            mGapbinN = mGapbinNs[0];
        else:
            print("ERROR: the bin sizes are inconsistent.");
            print("Run stops.")
            sys.exit(0);
        if len(np.unique(mGapRangeXs)) == 1:
            mGapRangeX = mGapRangeXs[0];
        else:
            print("ERROR: the x ranges are inconsistent.");
            print("Run stops.")
            sys.exit(0);
        mGapPDFss[J] = cJdf["PDF"].to_list();
        mGapNss[J]   = cJdf["dataPtN"].to_list();
        for i in range(J):
            mGapNss[J].insert(0, J-1-i);
##Yellin's C0 and CMAX
    if verbosity >= 1: print("Sampling the lower bound with signalN=" + str(signalN) + ":");
    nbins = np.linspace(rangeX[0], rangeX[1], binN+1);
    nbins = nbins[:-1];
    signalCDF = gausCDF_lambda(signalMu, signalSig);    #signal shape
    mGapMu = signalCDF(rangeG[1]) - signalCDF(rangeG[0]);
    cMAXsAlpha     = [ [] for _ in alphas ];
    cMAXsAlphaGood = [ [] for _ in alphas ];
    for s in tqdm(range(sampleN)):
        signalPDF = np.random.normal(signalMu, signalSig, signalN);
        #getting maximum gaps
        signalPDF = [x for x in signalPDF if ((rangeG[0]<=x) and (x<=rangeG[1]))];
        sortedPDF = np.sort(signalPDF).tolist();
        if incBoundary == True:
            sortedPDF = np.append(rangeG[0], sortedPDF);
            sortedPDF = np.append(sortedPDF, rangeG[1]);
        mGapSizes = [0]*(len(sortedPDF)-1);
        mGapIdxs  = [0]*(len(sortedPDF)-1);
        for leftIdx, leftPoint in enumerate(sortedPDF[:-1]):
            for i, rightPoint in enumerate(sortedPDF[leftIdx+1:]):
                gapSize = (signalCDF(rightPoint) - signalCDF(leftPoint));
                if gapSize > mGapSizes[i]:
                    mGapSizes[i] = gapSize;
                    mGapIdxs[i]  = leftIdx;
        #getting lower bounds 
        optFactor = 5.0;
        optRange  = [len(sortedPDF)/optFactor, optFactor*len(sortedPDF)];
        normUpBdMCs = [[pow(10.0, 6)]*(len(sortedPDF)-1) for _ in alphas]; 
        for J, mGapX in enumerate(mGapSizes):
            mGapPDFs    = mGapPDFss[J];
            mGapNs      = mGapNss[J];
            mGapR       = 1.0*mGapX/mGapMu;
            mGapRbinIdx = int(mGapbinN*(mGapR-mGapRangeX[0])/\
                                       (mGapRangeX[1]-mGapRangeX[0]));
            mGapCDFs_mGapR = [1.0]*len(mGapNs);
            for i in range(J):
                mGapCDFs_mGapR[i] = 0.0;
            for i in range(len(mGapPDFs)):
                mGapCDFs_mGapR[J+i] = mGapMCGetCDF(mGapbinN, mGapRangeX,\
                                                   mGapPDFs[i], mGapR);
                if abs(1.0 - mGapCDFs_mGapR[J+i]) < BREAKBOUND:
                    break;
            if verbosity >= 2:
                print("J =", J);
                print("   (x, mu, r) =", [mGapX, mGapMu, mGapR]); 
                print("   gap int    =", [sortedPDF[mGapIdxs[J]], 
                                          sortedPDF[mGapIdxs[J]+1+J]]);
                #print("  ", mGapCDFs_mGapR);
            for a, alpha in enumerate(alphas):
                gapProbMC    = \
                    mGapMCProb_lambda(mGapCDFs_mGapR,mGapNs,mGapMu);
                gapProbOptMC = \
                    mGapMCOpt_lambda( mGapCDFs_mGapR,mGapNs,mGapMu,alpha);
                normOptMC = optimize.minimize_scalar(gapProbOptMC,\
                                bounds=(optRange[0], optRange[1]),\
                                method="bounded", options={"xatol": LOWBOUND});
                normUpBdMCs[a][J] = normOptMC.x;
                if verbosity >= 2:
                    print("   (alpha, opt norm) =", [alpha, normUpBdMCs[a][J]]);
                    print("    alpha err    = 10e",\
                          np.log10(np.abs(alpha-gapProbMC(normUpBdMCs[a][J]))),\
                          sep="");
        for a, alpha in enumerate(alphas):
            minNormUpBdMC = min(normUpBdMCs[a]);
            cMAXsAlpha[a].append(minNormUpBdMC);
            if minNormUpBdMC >= signalN:
                cMAXsAlphaGood[a].append(minNormUpBdMC);
    if verbosity >= 2:
        for a, alpha in enumerate(alphas):
            print("(alpha, alphaJopt) =",\
                  [alpha, 1.0*len(cMAXsAlphaGood[a])/len(cMAXsAlpha[a])]);
            print("  ", cMAXsAlpha[a]);
#pickle save
    pathlib.Path("pickle").mkdir(exist_ok=True);
    pickleName = "pickle/maxGapOptIntAlpha.pickle";
    try:
        df = pd.read_pickle(pickleName);
    except (OSError, IOError) as e:
        columnNames=["signalN", "rangeL", "rangeU", "sampleN", "incBoundary", \
                     "alphaInput", "alphaJopt", "cMAXs"];
        df = pd.DataFrame(columns = columnNames);
    if verbosity >= 1:
        print("The following files have been updated:");
        print("    ", pickleName);
    for a, alpha in enumerate(alphas):
        bdData = {"signalN":     signalN, \
                  "rangeL":      rangeG[0],\
                  "rangeU":      rangeG[1],\
                  "sampleN":     sampleN, \
                  "incBoundary": incBoundary,\
                  "alphaInput":  alpha,\
                  "alphaJopt":   1.0*len(cMAXsAlphaGood[a])/len(cMAXsAlpha[a])};
        bdData["cMAXs"] = cMAXsAlpha[a];
        cond = (df["signalN"]==signalN) & (df["alphaInput"]==alpha);
        if df[cond].empty == False:
            df = df.drop(df[cond].index);
        df = df.append(bdData, ignore_index=True);
    if verbosity >= 1:
        print(df[df["signalN"] == signalN].head(len(alphas)));
    df.to_pickle(pickleName);
#hist
    signalHist = np.zeros(binN);
    for x in signalPDF: 
        if rangeX[0] < x and x < rangeX[1]:
            histIdx = int(np.floor(binN*(x-rangeX[0])/(rangeX[1]-rangeX[0])));
            signalHist[histIdx] += 1;
    nbinsNorm = np.linspace(rangeNorm[0], rangeNorm[1], binNNorm);
    cMAXHist = np.zeros((len(alphas), binNNorm));
    for a, alpha in enumerate(alphas):
        for norm in cMAXsAlpha[a]:
            if rangeNorm[0] < norm and norm < rangeNorm[1]:
                histIdx = int(np.floor(binNNorm*(norm-rangeNorm[0])/\
                                                (rangeNorm[1]-rangeNorm[0])));
                cMAXHist[a][histIdx] += 1;
#plots
    colors = ["orange", "purple", "red"];
    jump = int(np.ceil(len(alphas)/len(colors)));
    fig = plt.figure(figsize=(9, 7));
    gs = gridspec.GridSpec(1, 1);
    ax0 = fig.add_subplot(gs[0]);
    for c, color in enumerate(colors):
        ax0.plot(nbinsNorm, cMAXHist[min(c*jump, len(cMAXHist)-1)],\
                 linewidth=2, color=color, linestyle="steps-mid", \
                 alpha=(1.0-0.2*jump));
    ax0.set_title("Signal Upper Bound, CMAX under input alpha", \
                  fontsize=20, y=1.03);
    ax0.set_xlabel("norm upper bound", fontsize=18);
    ax0.set_ylabel("count", fontsize=18);
    ax0.set_xlim(rangeNorm[0], rangeNorm[1]);
    ax0.axhline(y=0,       xmin=0, xmax=1, color="black", linewidth=2);
    ax0.axvline(x=signalN, ymin=0, ymax=1, color="blue",  linewidth=2, alpha=0.5);
#save plots
    pathlib.Path("figure/maxGapOptIntAlpha").mkdir(parents=True, exist_ok=True);
    filenameFig = "figure/maxGapOptIntAlpha/alphaJoptS";
    filenameFig = filenameFig + str(signalN) + ".png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print("    ", filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




