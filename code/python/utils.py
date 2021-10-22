import math

def recursiveSplitGop(gopScores, delimiter="_"):
        names = list(gopScores.keys())
        lNames = names[0].split(delimiter)
        if len(lNames) == 1:
            return gopScores
        else:
            newGop = {}
            for n in names:
                name_splitted = n.split("_")
                if name_splitted[0] not in newGop:
                    newGop[name_splitted[0]] = {}
                newGop[name_splitted[0]][delimiter.join(name_splitted[1:])] = gopScores[n]

            newGop = {n:recursiveSplitGop(newGop[n]) for n in newGop}
            return newGop


def getUniqueFactors(gopScores, delimiter="_"):
    names = list(gopScores.keys())
    factors = [{n} for n in names[0].split(delimiter)]
    for n in names:
        for e, factor in enumerate(n.split(delimiter)):
            factors[e].add(factor)

    return factors

def chunk_based_on_number(lst, chunk_numbers):
    n = math.ceil(len(lst) / chunk_numbers)

    chunks = []
    for x in range(0, len(lst), n):
        each_chunk = lst[x: n + x]

        #if len(each_chunk) < n:
        #    each_chunk = each_chunk + [None for y in range(n - len(each_chunk))]
        chunks.append(each_chunk)

    return chunks