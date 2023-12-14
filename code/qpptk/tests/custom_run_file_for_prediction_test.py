import unittest
import os
import tempfile
from qpptk import main, parse_args
import pandas as pd
import numpy as np

class CustomRunFileForPredictionTest(unittest.TestCase):
    def test_with_run_file_where_one_query_is_highly_effective(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args(['-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/index/', '--run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run.txt', '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl', '--predPost', '--output', out_dir, '--cleanOutput'])
            main(args)

            actual = pd.read_json(out_dir + '/queries.jsonl', lines=True)

            self.assertEquals(len(actual), 3)
            # run is constructed so that query 1 is more effective than query 2 than query 3
            self.assertEquals(actual.iloc[0].to_dict()['qid'], 1.0)
            self.assertEquals(actual.iloc[0].to_dict()['wig+10'], 12.4250979385)
            self.assertEquals(actual.iloc[0].to_dict()['nqc+100'], 0.00043380560000000004)
            self.assertEquals(actual.iloc[0].to_dict()['smv+100'], -0.00043380560000000004)
            self.assertEquals(actual.iloc[0].to_dict()['clarity+1000+3'],  2.6993897409)

            self.assertEquals(actual.iloc[1].to_dict()['qid'], 2.0)
            self.assertEquals(actual.iloc[1].to_dict()['wig+10'], 4.2054528045)
            self.assertEquals(actual.iloc[1].to_dict()['nqc+100'], 0.2547211364)
            self.assertEquals(str(actual.iloc[1].to_dict()['smv+100']), 'nan')
            self.assertEquals(actual.iloc[1].to_dict()['clarity+1000+3'],  2.240679245)

            self.assertEquals(actual.iloc[2].to_dict()['qid'], 3.0)
            self.assertEquals(actual.iloc[2].to_dict()['wig+10'], -20.8716444259)
            self.assertEquals(actual.iloc[2].to_dict()['nqc+100'], 6.6373379689)
            self.assertEquals(str(actual.iloc[2].to_dict()['smv+100']), 'nan')
            self.assertEquals(actual.iloc[2].to_dict()['clarity+1000+3'],  3.2047245641)

    def test_with_run_file_where_all_queries_are_highly_effective(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args(['-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/index/', '--run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-02.txt', '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl', '--predPost', '--output', out_dir, '--cleanOutput'])
            main(args)

            actual = pd.read_json(out_dir + '/queries.jsonl', lines=True)

            self.assertEquals(len(actual), 3)
            # All queries should be predicted as rather effective
            self.assertEquals(actual.iloc[0].to_dict()['qid'], 1.0)
            self.assertEquals(actual.iloc[0].to_dict()['wig+10'], 12.4250979385)
            self.assertEquals(actual.iloc[0].to_dict()['nqc+100'], 0.00043380560000000004)
            self.assertEquals(actual.iloc[0].to_dict()['smv+100'], -0.00043380560000000004)
            self.assertEquals(actual.iloc[0].to_dict()['clarity+1000+3'],  2.6993897409)

            self.assertEquals(actual.iloc[1].to_dict()['qid'], 2.0)
            self.assertEquals(actual.iloc[1].to_dict()['wig+10'], 13.2004528045)
            self.assertEquals(actual.iloc[1].to_dict()['nqc+100'], 0.001559842)
            self.assertEquals(actual.iloc[1].to_dict()['smv+100'], -0.001559842)
            self.assertEquals(str(actual.iloc[1].to_dict()['clarity+1000+3']), 'nan')

            self.assertEquals(actual.iloc[2].to_dict()['qid'], 3.0)
            self.assertEquals(actual.iloc[2].to_dict()['wig+10'], 11.8874342352)
            self.assertEquals(actual.iloc[2].to_dict()['nqc+100'], 0.0007335282000000001)
            self.assertEquals(actual.iloc[2].to_dict()['smv+100'], -0.0007335281)
            self.assertEquals(str(actual.iloc[2].to_dict()['clarity+1000+3']), 'nan')
