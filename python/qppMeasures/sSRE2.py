import numpy as np

def sSRE2Single(qppColRanks, msrDFRanks, mapper):
    refMsr = mapper(qppColRanks.name)
    return ((qppColRanks - msrDFRanks[refMsr])/len(qppColRanks))**2


def sSRE2(msrDF, qppDF, rankType, mapper):
    '''

    Squared Ranking Error

    :param msrDF: Dataframe with the original measure
    :param qppDF: Dataframe with the qpp scores
    :param rankType: tiebreaking algorithm
    :param mapper: mapping function between qpp predictors and runs

    :return:
    '''

    #convert to ranks
    msrDFRanks = msrDF.rank(method=rankType)
    qppDFRanks = qppDF.rank(method=rankType)

    return qppDFRanks.apply(lambda x: sSRE2Single(x, msrDFRanks, mapper))




