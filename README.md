# Using Yellin's Maximum Gap Method to Estimate the Upper Count Limit of a Gaussian Signal Distribution under an Expoential Background Distribution from scratch

The code runs on python3 with additional packages:

    pip3 install scipy
    pip3 install tqdm
    ./macroRunOrderRef.sh
With the PYTHONPATH set correctly for `macroRunOrderRef.sh`. However, even then the above with take days to run, `macroRunOrderRef.sh` is more of a reference to guide the order of the python codes to run.
    
`mergePickle.py`: helper code for merging two pickle files to avoid having to run through the entire for loop (see `macroRunOrderRef.sh`) for each output reference pickle files.

`uniTransSampling.py`, `invTransSampling.py`: example codes for probability integral transform and inverse transform sampling to understand "distributed uniformly with unit density" to describe Eq.2 in Yellin's paper (<a href="https://arxiv.org/abs/physics/0203002">arxiv</a>).

`maxGapDistrGen.py`: Monte Carlo code that generates the probabily distribution $P(x, N, J)$: the probability of sampling $N$ points in a uniform distribution $y \in [0, 1]$, such that the maxium gap distance is $x = y_2 - y_1$ for any two points $y_1 < y_2$ among the $N+2$ points including the boundary 0 and 1, such that there are $J$ other points $y_i$ in between $[y_1, y_2]$. The output is saved in `pickle/maxGapDistr.pickle` for reference by the later codes. Turn on "testMode = True" to check out the code's functionality. This code is the most time consuming piece of the method, but it only needs to be done once.

`maxGapDistrDisp.py`: displaying the gap distribution $P(x, N, J)$ generated by `gapDistrGen.py`. The code output the following figures:

<kbd>
<img src="https://github.com/SphericalCowww/Stat_maximumGap_Yellin/blob/main/figureDisplay/gapDistrN10J0.png" width="800" height="300">
</kbd>
    
- Right: one of the 10M Monte Carlo sample from uniform distribution $y \in [0, 1]$ with $N=10$.
- Left: $P(x, N, J)$ for $J=0$, whose analysical form given by Eq.A4 in Yellin's is shown as the red curve.

<kbd>
<img src="https://github.com/SphericalCowww/Stat_maximumGap_Yellin/blob/main/figureDisplay/gapDistrN10J5.png" width="800" height="300">
</kbd>
    
- Left: $P(x, N, J)$ for $J=5$, which no longer has an analytical form.
    
`maxGapOptIntAlpha.py`: obtaining $C_n(x, \mu)$ from $P(x, N, J)$ generated by `gapDistrGen.py` by averaging over Poisson distribution (akin to Eq.A5 in Yellin's). Then Monta Carlo a given signal distribution with a given signal count $N_{sig}$ is used to find $C_0$ and $C_{max}$. In this case, the signal distribution is assume to be a known Gaussian with $\mu=0$ and $\sigma=1$. The output is saved in `pickle/maxGapOptIntAlpha.pickle` for reference by the later codes including $\alpha = 0.90, 0.93, 0.95, 0.96, 0.97, 0.98$.

`maxGapExp.py`: obtaining the upper count limit using $C_0$ and $C_{max}$ generated by `maxGapOptIntAlpha.py` for the same signal distribution. However, for this upper limit estimation, a simulated expoential background is also placed into the system but is assumed to be unknown to the estimation itself. The output is saved in `pickle/c0cMAXUpperBounds.pickle` for later comparison and generate the following figures:

<kbd>
<img src="https://github.com/SphericalCowww/Stat_maximumGap_Yellin/blob/main/figureDisplay/maxGapExpS10N0.png" width="800" height="300">
</kbd>

- Right: one of the 300 Monte Carlo samples from Gaussian distribution with $\mu=0$, $\sigma=1$, and $N_{sig}=10$.
- Left: with the 300 Monte Carlo samples, the green distribution is the $\alpha=0.9$ upper bounds of $N_{sig}$ estimated by $C_0$ and the orange by $C_{max}$. The vertial blue line indicate the actual $N_{sig}=10$.


<kbd>
<img src="https://github.com/SphericalCowww/Stat_maximumGap_Yellin/blob/main/figureDisplay/maxGapExpS10N10.png" width="800" height="300">
</kbd>

- Right: one of the 300 Monte Carlo samples from Gaussian distribution with $\mu=0$, $\sigma=1$, and $N_{sig}=10$ in blue. However, there are also a background exponential distribution with $\lambda=1$ and $N_{noise}=10$ in red. The estimator can only see the distribution as a combination of blue and red, has a knowledge of the blue distribution's shape but not the red distribution.

`poissonExp.py`: reading results generated by `maxGapOptIntAlpha.py` and compare it with the typical Poisson upper limit estimation method. The advantage of the maximum gap method is it does not require an "arbitrary" range as required by the Poisson method (as stated in Yellin's). The output is saved in `pickle/c0cMAXUpperBoundsPoisson.pickle` for later comparison and generate the following figures:

<kbd>
<img src="https://github.com/SphericalCowww/Stat_maximumGap_Yellin/blob/main/figureDisplay/maxGapExpS10N10Poisson.png" width="800" height="300">
</kbd>

- Left: with the 3000 Monte Carlo samples this time, the purple distribution is the $\alpha=0.9$ upper bounds of $N_{sig}$ estimated by Poisson method looking in range of [3.0, 5.0].

`upperBoundComparison.py`:
    
References:
- S. Yellin, Phys. Rev. D 66, 032005 (2002) (<a href="https://journals.aps.org/prd/abstract/10.1103/PhysRevD.66.032005">Phy Rev D</a>, <a href="https://arxiv.org/abs/physics/0203002">arxiv</a>)
- Wikipedia: Probability integral transform (<a href="https://en.wikipedia.org/wiki/Probability_integral_transform">wiki</a>)
- github: Inverse Transform Sampling (<a href="https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html">github</a>)
