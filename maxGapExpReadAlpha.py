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
def mGapMCGetProb(CDFs_x, Ns, L):
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
def mGapMCOpt_lambda(CDFs_x, Ns, mu, alpha):
    return lambda norm : abs(mGapMCGetProb(CDFs_x, Ns, norm*mu) - alpha);

###################################################################################
def main():
    if len(sys.argv) < 2:
        print("Please input the number of signal data points.");
        print("Run stops.")
        sys.exit(0);
    verbosity = 1;
    sampleN = 300;
    incBoundary = True;
    binN   = 1000;
    rangeX = [0.0, 5.0];
    rangeG = [1.0, 3.0];

    alpha = 0.9; 

    signalN = int(sys.argv[1]);
    #noiseN  = 0;                           #noise number
    noiseN  = 100;                   #noise number
    #np.random.seed(int(time.time()));
    np.random.seed(2);

    dataN   = noiseN + signalN;
    signalMu    = 2.0;
    signalSig   = 1.0;
    noiseLambda = 1.0;
    rangeNorm = [0.0, 100.0]; binNNorm = 100;
    if dataN > 60:
        rangeNorm = [0.0, 200.0]; binNNorm = 200; 
#dataframe from pickle
    if incBoundary == True: pickleName = "pickleRef/maxGapHists.pickle";
    else:                   pickleName = "pickleRef/maxGapHistsNoBd.pickle";
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

    dfAlpha = pd.read_pickle("pickle/MCerrRateAlpha.pickle");
    dfAlpha = dfAlpha[(dfAlpha["rangeL"] == rangeG[0] - signalMu)&\
                      (dfAlpha["rangeU"] == rangeG[1] - signalMu)];
    normFactor = stats.norm.cdf(rangeG[1]-signalMu)\
               - stats.norm.cdf(rangeG[0]-signalMu);
    dfSignalNMax = dfAlpha["signalN"].max();
##Yellin's C0 and CMAX
    if verbosity >= 1:
        print("Sampling the lower bound with signalN=" + str(signalN) + ":");
    nbins = np.linspace(rangeX[0], rangeX[1], binN+1);
    nbins = nbins[:-1];
    signalCDF = gausCDF_lambda(signalMu, signalSig);    #signal shape
    mGapMu = signalCDF(rangeG[1]) - signalCDF(rangeG[0]);
    c0s = [];
    cMAXs = [];
    for s in tqdm(range(sampleN)):
        noisePDF  = np.random.exponential(1.0/noiseLambda, noiseN);
        signalPDF = np.random.normal(signalMu, signalSig, signalN);
        dataPDF   = np.concatenate((noisePDF, signalPDF), axis=0);
        #getting maximum gaps
        dataPDF = [x for x in dataPDF if ((rangeG[0]<=x) and (x<=rangeG[1]))];
        sortedPDF = np.sort(dataPDF).tolist();
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
        
        optFactor = 5.0;
        optRange  = [len(sortedPDF)/optFactor, optFactor*len(sortedPDF)];
        #analytical
        mGapX  = mGapSizes[0];
        gapProbOptAna = mGapAnaOpt_lambda(mGapX, mGapMu, alpha);
        normOptAna = optimize.minimize_scalar(gapProbOptAna,
                        method="bounded", bounds=(optRange[0], optRange[1]),\
                        options={"xatol": LOWBOUND});
        gapProbAna = mGapAnaProb0_lambda(mGapX, mGapMu);
        normUpBdAna = normOptAna.x;
        c0s.append(normUpBdAna);
        if verbosity >= 2:
            print("S =", s);
            print("ana: opt norm        =", normUpBdAna);
            print("ana: (x, mu, k)      =", \
                  [mGapX, mGapMu, math.floor(1.0*mGapMu/mGapX)]);
            print("gap int              =", \
                  [sortedPDF[mGapIdxs[0]], sortedPDF[mGapIdxs[0]+1]]);
            print("ana: conf (set, opt) =", [alpha, gapProbAna(normUpBdAna)]);
        #Monte Carlo
        alphaRefSignalN = max(int(np.ceil(len(dataPDF)/normFactor)), dfSignalNMax);
        dfA = dfAlpha[dfAlpha["signalN"] == alphaRefSignalN];
        listA = dfA[["alphaInput", "alphaJopt"]].to_numpy();

        alphaJ = 1.0*alpha;
        if listA[0][1] < alpha:
            iterAU = 0;
            while listA[iterAU][1] < alpha:
                iterAU += 1;
                if iterAU >= len(listA):
                    print("WARNING: pickle/MCerrRateAlpha.pickle reaching\
                           alpha input upper bound at S=", alphaRefSignalN);
                    iterAU -= 1;
                    break;
            iterAL = iterAU - 1;
            slope = (listA[iterAU][0]-listA[iterAL][0])/\
                    (listA[iterAU][1]-listA[iterAL][1]);
            alphaJ = listA[iterAL][0] + slope*(alpha - listA[iterAL][1]); 
        normUpBdMCs = [pow(10.0, 6)]*(len(sortedPDF)-1);
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
            gapProbMC    = mGapMCProb_lambda(mGapCDFs_mGapR,mGapNs,mGapMu);
            gapProbOptMC = mGapMCOpt_lambda( mGapCDFs_mGapR,mGapNs,mGapMu,alphaJ);
            normOptMC = optimize.minimize_scalar(gapProbOptMC, 
                            method="bounded", bounds=(optRange[0], optRange[1]),\
                            options={"xatol": LOWBOUND});
            normUpBdMCs[J] = normOptMC.x;
            if verbosity >= 2:
                print("J =", J);
                print("MC:  opt norm        =", normUpBdMCs[J]);
                print("MC:  (x, mu, r)      =", [mGapX, mGapMu, mGapR]);
                print("MC:  gap int         =", [sortedPDF[mGapIdxs[J]], \
                                                 sortedPDF[mGapIdxs[J]+1+J]]);
                print("MC:  conf (set, opt) =", [alpha,\
                                                 gapProbMC(normUpBdMCs[J])]);
                print(mGapCDFs_mGapR);
        cMAX = min(normUpBdMCs);
        cMAXs.append(cMAX);
        if verbosity >= 2:
            print("");
            print("MC:  opt norm all J  =", cMAX);
    if verbosity >= 1:
        print("");
        print("c0:", c0s, "\n");
        print("cMAX:", cMAXs, "\n");
#pickle save
    pickleName = "c0cMAXHigherBounds.pickle";
    try:
        df = pd.read_pickle(pickleName);
    except (OSError, IOError) as e:
        columnNames=["signalN", "noiseN", "sampleN", "incBoundary",\
                     "c0s", "cMAXs"];
        df = pd.DataFrame(columns = columnNames);
    bdData = {"signalN":     signalN, \
              "noiseN":      noiseN, \
              "sampleN":     sampleN, \
              "incBoundary": incBoundary,\
              "c0s":         c0s,\
              "cMAXs":       cMAXs};
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
    ax1.plot(nbinsNorm, c0Hist,\
             linewidth=2, color="green",  linestyle="steps-mid");
    ax1.plot(nbinsNorm, cMAXHist,\
             linewidth=2, color="orange", linestyle="steps-mid");
    ax1.set_title("Signal Upper Bound from C0(green)&CMAX(orange)", \
                  fontsize=20, y=1.03);
    ax1.set_xlabel("norm upper bound", fontsize=18);
    ax1.set_ylabel("count", fontsize=18);
    ax1.set_xlim(rangeNorm[0], rangeNorm[1]);
    ax1.axhline(y=0, xmin=0, xmax=1, color="black", linewidth=2);
    ax1.axvline(x=signalN, ymin=0, ymax=1, color="blue", linewidth=2, alpha=0.5);
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__));
    filenameFig = exepath + "/dataFig/-maxGapRAS";
    filenameFig = filenameFig + str(signalN) + "N" + str(noiseN) + ".png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig);

    if verbosity >= 1:
        print("Creating the following files:");
        print("    ", filenameFig);

###################################################################################
if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




