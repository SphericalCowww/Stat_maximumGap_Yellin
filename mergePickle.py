import sys, math, re, time, os

import numpy as np
import pandas as pd
import pickle

#https://stackoverflow.com/questions/14128763
import difflib
def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

def main():
    pickleName1 = "c0cMAXHigherBounds1.pickle";
    pickleName2 = "c0cMAXHigherBounds2.pickle";
    df1 = pd.read_pickle(pickleName1);
    df2 = pd.read_pickle(pickleName2);

    pickleName = "--" + get_overlap(pickleName1, pickleName2) + "Merged.pickle";
    df = pd.concat([df1, df2]);
    df.to_pickle(pickleName);
    print("The merged pickle has been created:");
    print("    ", pickleName);

if __name__ == "__main__":
    print("\n##############################################################Head");
    main();
    print("##############################################################Tail");




