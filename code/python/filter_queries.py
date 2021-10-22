import argparse
import experimentalCollections.TrecCollections as tc
import pandas as pd
import numpy as np
import random

def idxmedian(d):
    ranks = d.rank(pct=True)
    close_to_median = abs(ranks - 0.5)
    return close_to_median.idxmin()

parser = argparse.ArgumentParser(description='Parse arguments for the META-framework.')

parser.add_argument('-c', '--collectionId', type=str,
                    help="collection to be used (i.e., RQV04).",
                    required=True, choices=['RQV04'])

parser.add_argument('-m', '--mLabel', type=str,
                    help="measure label (i.e., map).",
                    required=True, choices=['map'])

parser.add_argument('-rv', '--referenceVariant', type=str,
                    help="which variant to use to represent the topic",
                    choices=["title", "best", "worst", "median"])

parser.add_argument('-st', '--splitType', type=str,
                    help="type of split.",
                    choices=['qcut', 'cut'])


parser.add_argument('-ns', '--numberOfSplits', type=int,
                    help="measure label (i.e., map).",
                    choices=['map'])

'''
 only 1 variant per topic
 
 -rv --referenceVariant: title, best, worst 
 -st --splitType: quantile (absolute) 
 -ns --numberOfSplits: 3
 
 total: 5*6 + 3*6 + 2*6 = 60 -> too many anovas!!!
 
 more than 1 variant per topic
 - same as before, but with all the variants for each topic
'''

params = vars(parser.parse_args())


colData = getattr(tc, params["collectionId"])().importCollection(nThreads=15).evalRuns(params["mLabel"])

msrDF = pd.DataFrame.from_dict(colData.measure)
msrDF = pd.melt(msrDF.reset_index(), id_vars='index')
msrDF.columns = ["topic", "fullPipeline", "measure"]

queryOriginal = list(msrDF['topic'])
msrDF[["topic", "query"]] = msrDF['topic'].str.split("-", 1, expand=True)
msrDF["query"] = queryOriginal



means = msrDF.groupby(["query", "topic"]).mean().reset_index()



def getMedian(means):
    medians = means[['topic', 'measure']].groupby(['topic']).apply(idxmedian)
    medians = medians.drop("topic", axis=1)
    medians = medians.reset_index()
    medians['query'] = list(means['query'].iloc[list(medians['measure'])])
    medians['measure'] = list(means['measure'].iloc[list(medians['measure'])])

    return medians

def getBests(means):
    bests = means.groupby(['topic']).idxmax().reset_index()
    bests['query'] = list(means['query'].iloc[list(bests['measure'])])
    bests['measure'] = list(means['measure'].iloc[list(bests['measure'])])

    return bests

def getWorsts(means):
    worsts = means.groupby(['topic']).idxmin().reset_index()
    worsts['query'] = list(means['query'].iloc[list(worsts['measure'])])
    worsts['measure'] = list(means['measure'].iloc[list(worsts['measure'])])

    return worsts

def getTitles():
    tq = colData.getTitleQueries()
    titles = means[means['query'].isin(tq)].copy()

    return titles

fpath = "../../data/query_partitioning/querySplitting/"

fmapping = {"best": getBests(means), "median":getMedian(means), "worst":getWorsts(means), "title":getTitles()}
for nf, df in fmapping.items():
    for qt in [3, 4]:
        for fName, f in zip(["qcut", "cut"], [pd.qcut, pd.cut]):
            df[f'{qt}_{fName}'] = f(df['measure'], qt, labels=False)
            for iqt in range(qt):
                name = f"{nf}_{fName}_{iqt}_{qt}.csv"
                df[df[f'{qt}_{fName}']==iqt]['query'].to_csv(f"{fpath}{name}", header=["queries"], index=False)



queriesWithoutTitles = set(queryOriginal)-set(colData.getTitleQueries())
t2q = {}
for q in queriesWithoutTitles:
    t, _, _ = q.split("-")
    if t not in t2q:
        t2q[t] = []
    t2q[t].append(q)
lengths = [len(t2q[t]) for t in t2q]
minlength = np.min(lengths)

sampledQ = []
for t in t2q:
    sampledQ += random.sample(t2q[t], minlength)

sampledQ+=colData.getTitleQueries()

with open(f"{fpath}/sampledFormsAndTitles.csv", "w") as F:
    F.write("\n".join(["queries"]+sampledQ))

newMsrDF = means[means['query'].isin(sampledQ)].copy()
for qt in [3, 4]:
    f = lambda x: pd.qcut(x, qt, labels=False, duplicates='drop')
    newMsrDF['bucket'] = means[['topic', 'measure']].groupby("topic")['measure'].transform(f)
    for iqt in range(qt):
        name = f"variants_{iqt}_{qt}.csv"
        newMsrDF[newMsrDF['bucket'] == iqt]['query'].to_csv(f"{fpath}{name}", header=["queries"], index=False)
