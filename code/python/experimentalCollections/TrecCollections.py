import pytrec_eval
import time
import os
from utils import chunk_based_on_number
from multiprocessing import Pool


class AbstractCollection:

    def __init__(self, logger=None):
        self.logger = logger

    def importCollection(self, nThreads=1, selected_runs=None):
        if not self.logger is None:
            self.logger.info("importing the collection")
            stime = time.time()
        self.runs = self.import_runs(nThreads, selected=selected_runs)
        self.qrels = self.import_qrels()

        self.systems = list(self.runs.keys())
        self.topics = list(self.qrels.keys())

        if not self.logger is None:
            self.logger.info(f"collection imported in {time.time() - stime:.2f} seconds")

        return self

    def import_qrels(self, qPath=None):
        # -------------------------- IMPORT QRELS -------------------------- #
        if qPath is None:
            qPath = self.qrel_path

        with open(qPath, "r") as F:
            qrels = pytrec_eval.parse_qrel(F)

        return qrels

    def import_runs(self, nThreads, selected=None):

        systems_paths = os.listdir(self.runs_path)
        if selected is not None:
            systems_paths = [s for s in systems_paths if s.split(".")[0] in selected]
        # -------------------------- IMPORT RUNS -------------------------- #
        runs = {}
        chunks = chunk_based_on_number(systems_paths, nThreads)

        with Pool(processes=nThreads) as pool:

            futuresRunsDict = [pool.apply_async(getPartialRuns, [chunk, self.runs_path]) for chunk in chunks]
            runsDict = [res.get() for res in futuresRunsDict]

        for d in runsDict:
            for r in d:
                runs[r] = d[r]

        return runs

    def evalRuns(self, mLabel):
        if not self.logger is None:
            self.logger.info("computing measures...")
            stime = time.time()
        topic_evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, {mLabel})
        self.measure = {s: topic_evaluator.evaluate(self.runs[s]) for s in self.systems}

        # ---- remove the measure keyword from the measure dictionary
        self.measure = {r: {t: self.measure[r][t][mLabel] for t in self.measure[r]} for r in self.measure}

        if not self.logger is None:
            self.logger.info(f"done in {time.time() - stime:.2f} seconds")

        return self


class RQV04(AbstractCollection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_path = "../../../21-ECIR-CFFSZ/data/experiment/"
        self.runs_path = self.data_path + "runs/RQV04/"
        self.qrel_path = self.data_path + "pool/RQV04/expanded_robust04.qrels"

    def getTitleQueries(self):
        queries = list(self.import_qrels().keys())
        titles = []
        for q in queries:
            _, g, u = q.split("-")
            if g == "50" and u == "1":
                titles.append(q)

        return titles


# used to parallelize the import of the runs
def getPartialRuns(rNamesList, runs_path):
    runs = {}
    for e, run_filename in enumerate(rNamesList):
        with open(runs_path + run_filename, "r") as F:
            try:
                runs[run_filename.split(".")[0]] = pytrec_eval.parse_run(F)
            except Exception as e:
                print(e)
                print(run_filename)
    return runs
