from . import AbstractExperiment
import experimentalCollections.TrecCollections as tc
import experimentalCollections.qppCollections as qc
from utils import recursiveSplitGop, getUniqueFactors
import pandas as pd

import qppMeasures
import commonParams
import time

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import gaussian_kde

import matplotlib

#matplotlib.rcParams['text.usetex'] = True
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

#plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
class CompareMeasures(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):
        #dashes = ['-', '--', '-.', ':', (5, 2)]
        dashes =[(None, None), (1, 1), (4, 3), (3, 1, 1, 1), (2, 2)]
        tieStrategies = ['average', 'min', 'max', 'first', 'dense']

        dss = [pd.read_csv(f"../../data/distributional_measures/DistributionalMeasureTruncated_{self.collectionId}_{self.mLabel}_{self.distrMeasure}_{ts}_titleQueries.csv") for ts in tieStrategies]

        for e, ds in enumerate(dss):
            ds = ds[(ds['stemmer']=='porter') & (ds['stoplist']=='indri')]
            dss[e] = ds


        #msrs = [gaussian_kde(np.array(ds[self.distrMeasure]), bw_method='silverman')for ds in dss]

        #x = np.linspace(0, 1, 10000)

        #ests = [f(x) for f in msrs]

        #plt.figure()
        #for i, est in enumerate(ests):
        #    plt.plot(x, est, lw=2, label=tieStrategies[i], dashes=dashes[i])
        #plt.legend()
        #plt.show()


        fig, axs = plt.subplots(1, 5, figsize=(30, 8))
        for i, est in enumerate(dss):
            ax = axs[i]
            c, br = np.histogram(est[self.distrMeasure], bins=40, density=True)
            ax.plot(br[:-1]+(br[1:]-br[:-1])/2, c, label=rf"{self.distrMeasure}")


            means = est[['predictor', 'sARE']].groupby('predictor').mean().reset_index()
            f = gaussian_kde(np.array(means[self.distrMeasure]), bw_method='silverman')
            x = np.linspace(br[0], br[-1], 10000)
            y = f(x)

            ax.plot(x, y, label=rf"{self.distrMeasure.replace('s', 'sM')}")
            ax.legend(fontsize=24)
            ax.tick_params(labelsize=22)
            ax.set_title(rf"{tieStrategies[i]}", fontsize=24)
            ax.set_ylabel(r"density", fontsize=24)

        #fig.tight_layout(pad=1.0)
        plt.savefig("comparison-tiebreaking.pdf")



        measures = ['sARE', 'sRE', 'sRSRE', 'sSR']
        dss = [pd.read_csv(f"../../data/distributional_measures/DistributionalMeasure_{self.collectionId}_{self.mLabel}_{ms}_average_titleQueries.csv") for ms in measures]

        for e, ds in enumerate(dss):
            ds = ds[(ds['stemmer']=='porter') & (ds['stoplist']=='indri')]
            dss[e] = ds
            print(dss[e][measures[e]])


        #msrs = [gaussian_kde(np.array(dss[e][ms]), bw_method=0.1) for e, ms in enumerate(measures)]

        #x = np.linspace(-1.5, 3, 10000)

        #ests = [f(x) for f in msrs]


        fig, axs = plt.subplots(1, 4, figsize=(30, 8))
        for i, est in enumerate(dss):
            ax = axs[i]
            c, br = np.histogram(est[measures[i]], bins=40, density=True)
            ax.plot(br[:-1]+(br[1:]-br[:-1])/2, c, label=rf"{measures[i]}")


            means = est[['predictor', measures[i]]].groupby('predictor').mean().reset_index()
            f = gaussian_kde(np.array(means[measures[i]]), bw_method='silverman')
            x = np.linspace(br[0], br[-1], 10000)
            y = f(x)

            ax.plot(x, y, label=rf"{measures[i].replace('s', 'sM')}")
            ax.legend(fontsize=24)
            ax.tick_params(labelsize=22)
            ax.set_ylabel(r"density", fontsize=24)


        plt.savefig("comparison-distrMeasures.pdf")



