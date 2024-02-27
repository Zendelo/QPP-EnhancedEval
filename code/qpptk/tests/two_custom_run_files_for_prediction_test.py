import unittest
import os
import tempfile
from pathlib import Path
import pandas as pd
from qpptk import replace_scores_in_run_file_with_reference_scores, parse_args, main
from approvaltests import verify_file

class TwoCustomRunFilesForPredictionTest(unittest.TestCase):
    def test_unification_of_run_files_run_01(self):
        unified_run_file = replace_scores_in_run_file_with_reference_scores(
            run_file=Path(os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run.txt'),
            reference_run_file=Path(os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-with-ground-truth-scores.txt')
        )

        verify_file(unified_run_file)

    def test_unification_of_run_files_run_02(self):
        unified_run_file = replace_scores_in_run_file_with_reference_scores(
            run_file=Path(os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-02.txt'),
            reference_run_file=Path(os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-with-ground-truth-scores.txt')
        )

        verify_file(unified_run_file)


    def test_end_to_end_with_run_01(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/index/',
                '--run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run.txt',
                '--use-scores-from-run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-with-ground-truth-scores.txt',
                '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl',
                '--predPost', '--cleanOutput',
                '--output', out_dir,
                '--stats_index_path', stats_dir
            ])
            main(args)

            actual = pd.read_json(out_dir + '/queries.jsonl', lines=True)

            self.assertEqual(len(actual), 3)
            # run is constructed so that query 1 is more effective than query 2 than query 3
            self.assertEqual(actual.iloc[0].to_dict()['qid'], 1.0)
            self.assertEqual(actual.iloc[0].to_dict()['wig+10'], 6.9431571325)
            self.assertEqual(actual.iloc[0].to_dict()['nqc+100'], 0.043380556800000004)
            self.assertEqual(actual.iloc[0].to_dict()['smv+100'], -0.01)
            self.assertEqual(actual.iloc[0].to_dict()['clarity+1000+100'], 2.7431934039)

            self.assertEqual(actual.iloc[1].to_dict()['qid'], 2.0)
            self.assertEqual(actual.iloc[1].to_dict()['wig+10'], 3.8721194712)
            self.assertEqual(actual.iloc[1].to_dict()['nqc+100'], 0.1470633166)
            self.assertEqual(actual.iloc[1].to_dict()['smv+100'], -0.01)
            self.assertEqual(actual.iloc[1].to_dict()['clarity+1000+100'], 2.3208312544)

            self.assertEqual(actual.iloc[2].to_dict()['qid'], 3.0)
            self.assertEqual(actual.iloc[2].to_dict()['wig+10'], 4.8199019572)
            self.assertEqual(actual.iloc[2].to_dict()['nqc+100'], 0.0)
            self.assertEqual(actual.iloc[2].to_dict()['smv+100'], -0.01)
            self.assertEqual(actual.iloc[2].to_dict()['clarity+1000+100'],  2.3616150510000002)

    def test_end_to_end_with_run_01_approval(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/index/',
                '--run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run.txt',
                '--use-scores-from-run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-with-ground-truth-scores.txt',
                '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl',
                '--predPost', '--cleanOutput',
                '--output', out_dir,
                '--stats_index_path', stats_dir
            ])
            main(args)

            verify_file(out_dir + '/queries.jsonl')

    def test_end_to_end_with_run_02(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/index/',
                '--run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-02.txt',
                '--use-scores-from-run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-with-ground-truth-scores.txt',
                '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl',
                '--predPost', '--cleanOutput',
                '--output', out_dir,
                '--stats_index_path', stats_dir
            ])
            main(args)

            actual = pd.read_json(out_dir + '/queries.jsonl', lines=True)

            self.assertEqual(len(actual), 3)
            # All queries should be predicted as rather effective
            self.assertEqual(actual.iloc[0].to_dict()['qid'], 1.0)
            self.assertEqual(actual.iloc[0].to_dict()['wig+10'], 6.9431571325)

            self.assertEqual(actual.iloc[1].to_dict()['qid'], 2.0)
            self.assertEqual(actual.iloc[1].to_dict()['wig+10'], 3.7054528045)

            self.assertEqual(actual.iloc[2].to_dict()['qid'], 3.0)
            self.assertEqual(actual.iloc[2].to_dict()['wig+10'], 5.1734553478)


    def test_end_to_end_with_run_02_approval(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/index/',
                '--run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-02.txt',
                '--use-scores-from-run-file', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/run-with-ground-truth-scores.txt',
                '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl',
                '--predPost', '--cleanOutput',
                '--output', out_dir,
                '--stats_index_path', stats_dir
            ])
            main(args)
            verify_file(out_dir + '/queries.jsonl')
