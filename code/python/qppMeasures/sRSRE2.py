import numpy as np

def sRSRE2Single(qppColRanks, msrDFRanks, mapper):
    refMsr = mapper(qppColRanks.name)
    return np.sqrt((qppColRanks - msrDFRanks[refMsr])**2)/len(qppColRanks)


def sRSRE2(msrDF, qppDF, rankType, mapper):
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

    return qppDFRanks.apply(lambda x: sRSRE2Single(x, msrDFRanks, mapper))




