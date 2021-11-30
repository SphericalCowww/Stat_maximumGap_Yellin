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
    
def main():
    verbosity = 1;
    incBoundary = True;
#reading pickle
    pickleName = "c0cMAXHigherBounds.pickle";
    df = pd.read_pickle(pickleName);
    df = df[df["noiseN"] != 0];
    df = df.drop(df[df["signalN"] == 0].index);
    df = df.sort_values("signalN");
    signalNs = df["signalN"].to_numpy();
    c0s      = df["c0s"].to_numpy();
    cMAXs    = df["cMAXs"].to_numpy();
    c0Meds   = [];
    cMAXMeds = [];
    c0Errs   = [];
    cMAXErrs = [];
    for i, sigN in enumerate(signalNs):
        c0Meds.append(np.median(c0s[i])/sigN);
        cMAXMeds.append(np.median(cMAXs[i])/sigN);
        c0Errs.append(  len(np.where(np.array(c0s[i])<1.0*sigN)[0])\
                       /len(c0s[i]));
        cMAXErrs.append(len(np.where(np.array(cMAXs[i])<1.0*sigN)[0])\
                       /len(cMAXs[i]));
    try:
        poissons = df["poissons"].to_numpy();
    except KeyError:
        poissons = None;
    poiMeds = [];
    poiErrs = [];
    if poissons is not None:
        for i, sigN in enumerate(signalNs):
            poiMeds.append(np.median(poissons[i])/sigN);
            poiErrs.append(len(np.where(np.array(poissons[i])<1.0*sigN)[0])\
                          /len(poissons[i]));
    if verbosity >= 1:
        print("Signal data points from:");
        print(signalNs);
#plots
    fig = plt.figure(figsize=(9, 10));
    gs = gridspec.GridSpec(2, 1);
    ax0 = fig.add_subplot(gs[0]);
    ax1 = fig.add_subplot(gs[1]);


    if poissons is not None:
        ax0.plot(signalNs, poiMeds, linewidth=2.5, color="orange");
        ax1.plot(signalNs, poiErrs, linewidth=2.5, color="orange");
        ax0.set_title("Signal Higher Bound, C0(green)&CMAX(purple)&Poi(orange)",\
                      fontsize=16, y=1.03);
        ax1.set_title("Bound Error Ratio, C0(green)&CMAX(purple)&Poi(orange)",\
                      fontsize=16, y=1.03);
    else:
        ax0.set_title("Signal Higher Bound, C0(green)&CMAX(purple)", \
                      fontsize=20, y=1.03); 
        ax1.set_title("Bound Error Ratio, C0(green)&CMAX(purple)", \
                      fontsize=20, y=1.03); 
    #plot 0
    ax0.plot(signalNs, c0Meds, linewidth=2, color="green");
    ax0.plot(signalNs, cMAXMeds, linewidth=2, color="purple");
    ax0.set_xlabel("true signal count", fontsize=18);
    ax0.set_ylabel("bound/true count ratio", fontsize=18);
    ax0.set_xlim(0, 60);
    ax0.set_ylim(1, 5);
    #plot 1
    ax1.plot(signalNs, c0Errs, linewidth=2, color="green");
    ax1.plot(signalNs, cMAXErrs, linewidth=2, color="purple");
    ax1.set_xlabel("true signal count", fontsize=18);
    ax1.set_ylabel("error ratio", fontsize=18);
    ax1.set_xlim(0, 60);
    ax1.set_ylim(0, 0.3);
#save plots
    gs.tight_layout(fig);
    exepath = os.path.dirname(os.path.abspath(__file__));
    filenameFig = exepath+"/fig/c0cMAXHigherBound.png";
    plt.savefig(filenameFig);
    if verbosity >= 1:
        print("Creating the following files:");
        print("    ", filenameFig);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




