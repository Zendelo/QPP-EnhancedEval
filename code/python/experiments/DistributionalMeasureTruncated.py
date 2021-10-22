from . import AbstractExperiment
import experimentalCollections.TrecCollections as tc
import experimentalCollections.qppCollections as qc
from utils import recursiveSplitGop, getUniqueFactors
import pandas as pd

import qppMeasures
import commonParams
import time

class DistributionalMeasureTruncated(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):

        globalstime = time.time()
        colData = getattr(tc, self.collectionId)(logger=self.logger).importCollection(nThreads=self.processors).evalRuns(self.mLabel)
        qppData = getattr(qc, self.collectionId)(logger=self.logger).importQPPScores()


        qppDF = pd.DataFrame.from_dict(qppData.scores)
        msrDF = pd.DataFrame.from_dict(colData.measure)

        qppDF = qppDF.round(2)

        if hasattr(self, 'queryPartitioning'):
            selected_queries = pd.read_csv(f"../../data/query_partitioning/querySplitting/{self.queryPartitioning}.csv")
            qppDF = qppDF[qppDF.index.isin(selected_queries['queries'])]

        self.logger.info(f"computing the {self.distrMeasure} measure")
        stime = time.time()
        # select only queries present in the qpp
        msrDF = msrDF[msrDF.index.isin(qppDF.index)]
        
        mapper = getattr(qppMeasures.mappers, self.collectionId)

        distrMeas = getattr(qppMeasures, self.distrMeasure)(msrDF, qppDF, self.rankType, mapper)

        self.logger.info(f"done in {time.time()-stime:.2f} seconds")


        self.logger.info(f"saving...")
        stime = time.time()
        distrMeas = pd.melt(distrMeas.reset_index(), id_vars='index')
        distrMeas.columns = ["topic", "fullPipeline", self.distrMeasure]
        queryOriginal = list(distrMeas['topic'])
        distrMeas[["topic", "query"]] = distrMeas['topic'].str.split("-", 1, expand=True)
        distrMeas["query"] = queryOriginal
        distrMeas[getattr(commonParams, f"{self.collectionId}Factors")] = distrMeas['fullPipeline'].str.split("_", expand=True)



        distrMeas.to_csv(f"../../data/distributional_measures/{self.expLabel}.csv",index=False)
        self.logger.info(f"done in {time.time() - stime:.2f} seconds")

        self.logger.info(f"experiment successfully terminated. done in {time.time() - globalstime:.2f} seconds.")

