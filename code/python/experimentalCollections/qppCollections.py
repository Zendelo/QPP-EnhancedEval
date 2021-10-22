import os

class abstractQppCollection:

    def __init__(self, logger=None):
        self.logger = logger

    def importQPPScores(self):
        systems_paths = os.listdir(self.qpps_path)
        # -------------------------- IMPORT RUNS -------------------------- #
        self.scores = {}
        for e, qpps_filename in enumerate(systems_paths):
            qppName = qpps_filename.split(".")[0]
            self.scores[qppName] = {}
            with open(self.qpps_path + qpps_filename, 'r') as fp:
                lines = fp.readlines()
                for l in lines:
                    _, t, s = l.strip().split()
                    self.scores[qppName][t] = float(s)

        return self




class RQV04(abstractQppCollection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_path = "../../../21-ECIR-CFFSZ/data/experiment/"
        self.qpps_path = self.data_path + "qpp_raw/RQV04/"

