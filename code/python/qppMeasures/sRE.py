import numpy as np

def sRESingle(qppColRanks, msrDFRanks, mapper):
    refMsr = mapper(qppColRanks.name)
    return (qppColRanks - msrDFRanks[refMsr])/len(qppColRanks)


def sRE(msrDF, qppDF, rankType, mapper):


    #convert to ranks
    msrDFRanks = msrDF.rank(method=rankType)
    qppDFRanks = qppDF.rank(method=rankType)

    return qppDFRanks.apply(lambda x: sRESingle(x, msrDFRanks, mapper))




