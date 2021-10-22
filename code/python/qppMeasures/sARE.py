import numpy as np

def sARESingle(qppColRanks, msrDFRanks, mapper):
    refMsr = mapper(qppColRanks.name)

    return (np.abs(qppColRanks - msrDFRanks[refMsr])/len(qppColRanks))


def sARE(msrDF, qppDF, rankType, mapper):


    #convert to ranks
    msrDFRanks = msrDF.rank(method=rankType)
    qppDFRanks = qppDF.rank(method=rankType)

    return qppDFRanks.apply(lambda x: sARESingle(x, msrDFRanks, mapper))




