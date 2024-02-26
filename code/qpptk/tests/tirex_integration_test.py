import unittest
from tira.rest_api_client import Client
from qpptk import main, parse_args
from approvaltests import verify_file
import tempfile

class TirexIntegrationTest(unittest.TestCase):
    def test_on_cranfield_dataset_with_approvaltests(self):
        tira = Client()
        index_dir = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', 'cranfield-20230107-training') + '/index'
        queries = tira.download_dataset('ir-benchmarks', 'cranfield-20230107-training') + '/queries.jsonl'

        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', index_dir,
                '--jsonl_queries', queries,
                '--output', out_dir,
                '--stats_index_path',  stats_dir,
                '--predict', '--retrieve', '--cleanOutput'
            ])
            main(args)

            # I only spot-checked that the output looks reasonable, no in-depth tests
            verify_file(out_dir + '/queries.jsonl')

    def test_on_cranfield_dataset_with_non_string_query_ids_approvaltests(self):
        tira = Client()
        index_dir = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', 'cranfield-20230107-training') + '/index'
        queries = 'tests/resources/small-example-03/queries.jsonl'

        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', index_dir,
                '--jsonl_queries', queries,
                '--output', out_dir,
                '--stats_index_path',  stats_dir,
                '--predict', '--retrieve', '--cleanOutput'
            ])
            main(args)

            # I only spot-checked that the output looks reasonable, no in-depth tests
            verify_file(out_dir + '/queries.jsonl')

    def test_on_cranfield_dataset_with_non_string_query_ids_and_oov_queries_approvaltests(self):
        tira = Client()
        index_dir = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', 'cranfield-20230107-training') + '/index'
        queries = 'tests/resources/small-example-04/queries.jsonl'

        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', index_dir,
                '--jsonl_queries', queries,
                '--output', out_dir,
                '--stats_index_path',  stats_dir,
                '--predict', '--retrieve', '--cleanOutput'
            ])
            main(args)

            # I only spot-checked that the output looks reasonable, no in-depth tests
            verify_file(out_dir + '/queries.jsonl')

    def test_on_cranfield_dataset_with_approvaltests_and_bm25_run(self):
        tira = Client()
        index_dir = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', 'cranfield-20230107-training') + '/index'
        bm25_run = tira.get_run_output('ir-benchmarks/tira-ir-starter/BM25 Re-Rank (tira-ir-starter-pyterrier)', 'cranfield-20230107-training') + '/run.txt'
        queries = tira.download_dataset('ir-benchmarks', 'cranfield-20230107-training') + '/queries.jsonl'

        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', index_dir,
                '--jsonl_queries', queries,
                '--run-file', bm25_run,
                '--output', out_dir,
                '--stats_index_path',  stats_dir,
                '--predPost', '--retrieve', '--cleanOutput'
            ])
            main(args)

            # I only spot-checked that the output looks reasonable, no in-depth tests
            verify_file(out_dir + '/queries.jsonl')

    def test_on_cranfield_dataset_with_approvaltests_and_bm25_run_with_monot5_scores(self):
        tira = Client()
        index_dir = tira.get_run_output('ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)', 'cranfield-20230107-training') + '/index'
        bm25_run = tira.get_run_output('ir-benchmarks/tira-ir-starter/BM25 Re-Rank (tira-ir-starter-pyterrier)', 'cranfield-20230107-training') + '/run.txt'
        monot5_run = tira.get_run_output('ir-benchmarks/tira-ir-starter/MonoT5 Base (tira-ir-starter-gygaggle)', 'cranfield-20230107-training') + '/run.txt'
        queries = tira.download_dataset('ir-benchmarks', 'cranfield-20230107-training') + '/queries.jsonl'

        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            args = parse_args([
                '-ti', index_dir,
                '--jsonl_queries', queries,
                '--run-file', bm25_run,
                '--use-scores-from-run-file', monot5_run,
                '--output', out_dir,
                '--stats_index_path',  stats_dir,
                '--predPost', '--retrieve', '--cleanOutput'
            ])
            main(args)

            # I only spot-checked that the output looks reasonable, no in-depth tests
            verify_file(out_dir + '/queries.jsonl')
