def RQV04(nQppModel, delimiter ="_"):

    _, _, stop, stem, _, _ = nQppModel.split(delimiter)

    return f"robust04_{stop}_{stem}_QL"