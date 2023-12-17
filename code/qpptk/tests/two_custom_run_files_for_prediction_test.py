#multiple run files

import unittest
import os
import tempfile
from pathlib import Path
from qpptk import replace_scores_in_run_file_with_reference_scores
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
