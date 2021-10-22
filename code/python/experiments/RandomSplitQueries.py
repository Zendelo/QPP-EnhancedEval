from . import AbstractExperiment
import experimentalCollections.TrecCollections as tc
import experimentalCollections.qppCollections as qc
from utils import recursiveSplitGop, getUniqueFactors
import pandas as pd

import qppMeasures
import commonParams
import time
import numpy as np
import random

class RandomSplitQueries(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):
        globalstime = time.time()
        colData = getattr(tc, self.collectionId)(logger=self.logger).importCollection(
            nThreads=self.processors).evalRuns(self.mLabel)


        t2f = {}
        for fId in colData.qrels.keys():
            tId = fId.split("-")[0]
            if tId not in t2f:
                t2f[tId] = set()
            t2f[tId].add(fId)


        keptFormulations = []
        if (self.queryPartitioning=='maxCommonWithTitle'):
            keptNumber = np.min([len(forms) for _, forms in t2f.items()])

            for tId in t2f:
                if "-".join(tId.split("-")[1:])=="50-1":
                    keptFormulations.append(tId)

        else:
            keptNumber = int(self.queryPartitioning)
            for tId in t2f:
                keptFormulations+=random.sample(t2f[tId], keptNumber)

        with open(f"../../data/query_partitioning/querySplitting/{self.queryPartitioning}.csv", "w") as F:
            F.write("queries\n")
            for f in keptFormulations:
                F.write(f"{f}\n")


