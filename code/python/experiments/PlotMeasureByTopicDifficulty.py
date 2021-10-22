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
import matplotlib.font_manager

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

class PlotMeasureByTopicDifficulty(AbstractExperiment):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def run_experiment(self):


        colData = getattr(tc, self.collectionId)(logger=self.logger).importCollection(
            nThreads=self.processors, selected_runs='robust04_indri_porter_QL').evalRuns(self.mLabel)

        oMsrDF = pd.DataFrame.from_dict(colData.measure)

        selectedQueries = colData.getTitleQueries()
        oMsrDF = oMsrDF[oMsrDF.index.isin(selectedQueries)]
        msrDF = pd.melt(oMsrDF.reset_index(), id_vars='index')
        msrDF.columns = ["topic", "fullPipeline", "measure"]

        queryOriginal = list(msrDF['topic'])
        msrDF[["topic", "query"]] = msrDF['topic'].str.split("-", 1, expand=True)
        msrDF["query"] = queryOriginal

        cuts = ['hard', 'medium', 'easy']
        msrDF['difficulty'] = pd.qcut(msrDF['measure'], len(cuts), labels=cuts)

        qppData = getattr(qc, self.collectionId)(logger=self.logger).importQPPScores()
        qppDF = pd.DataFrame.from_dict(qppData.scores)
        qppDF = qppDF[[c for c in qppDF.columns if "indri_porter" in c]]

        msrs = []

        for m in cuts:
            qrs = msrDF[msrDF["difficulty"]==m]["query"]
            mapper = getattr(qppMeasures.mappers, self.collectionId)
            distrMeas = getattr(qppMeasures, self.distrMeasure)(oMsrDF[oMsrDF.index.isin(qrs)], qppDF[qppDF.index.isin(qrs)], self.rankType, mapper)
            distrMeas = pd.melt(distrMeas.reset_index(), id_vars='index')
            distrMeas.columns = ["topic", "fullPipeline", self.distrMeasure]
            queryOriginal = list(distrMeas['topic'])
            distrMeas[["topic", "query"]] = distrMeas['topic'].str.split("-", 1, expand=True)
            distrMeas["query"] = queryOriginal
            distrMeas[getattr(commonParams, f"{self.collectionId}Factors")] = distrMeas['fullPipeline'].str.split("_", expand=True)
            distrMeas['difficulty'] = [m for _ in range(len(distrMeas))]
            msrs.append(distrMeas)

        df = pd.concat(msrs)

        fs = 18

        plt.figure(figsize=(12,10))
        sns.set_context("paper", rc={"font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs})
        order = ['wig', 'smv', 'nqc', 'clarity',
                 'uef-wig', 'uef-smv', 'uef-nqc', 'uef-clarity',]
        g = sns.boxplot(data=df[df['retrievalFunction']=='QL'], x='predictor', y=self.distrMeasure, hue="difficulty", order=order)
        plt.xticks(np.arange(8), labels=['WIG', 'SMV', 'NQC', 'Clarity', 'UEF(WIG)',  'UEF(SMV)', 'UEF(NQC)', 'UEF(Clarity)'], rotation=45, fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.legend(fontsize=fs)
        plt.savefig("distr_difficulty_QL.pdf")



        #plt.figure(figsize=(42, 22))
        plt.figure(figsize=(12, 10))
        sns.set_context("paper", rc={"font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs})
        order = ['avg-idf', 'max-idf', 'var', 'avg-var', 'max-var', 'scq', 'avg-scq', 'max-scq']
        g = sns.boxplot(data=df[df['retrievalFunction']=='PRE'], x='predictor', y=self.distrMeasure, hue="difficulty", order=order)
        plt.xticks(np.arange(8), labels=[r"AvgIDF", r"MaxIDF", r"VAR", r"AvgVAR", r"MaxVAR",  r"SCQ", r"AvgSCQ", r"MaxSCQ"], rotation=45, fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.legend(fontsize=fs)
        #g.legend(new_labels, fontsize=24)
        plt.savefig("distr_difficulty_PRE.pdf")
        '''
        for  mid in range(n_cuts):
            tmpDf = df[df['difficulty']==mid]
            aggTmpDf = df.groupby('predictor')[self.distrMeasure].agg(['mean'])
            aggTmpDf = aggTmpDf.reset_index()
            plt.figure()
            sns.boxplot(data=tmpDf, x='predictor', y=self.distrMeasure, order=aggTmpDf.sort_values('mean').predictor)
            plt.xticks(rotation=45)
            plt.show()
        '''