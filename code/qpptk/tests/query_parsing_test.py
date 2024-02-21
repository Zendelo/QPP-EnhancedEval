import unittest
from qpptk import get_queries_object, parse_args
import tempfile
import os


class QueryParsingTest(unittest.TestCase):
    def test_end_to_end_with_porter_stemmer(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            expected = [
                {'achiev': 1, 'hubbl': 1, 'telescop': 1}, 
                {'exit': 1, 'how': 1, 'to': 1, 'vim': 1},
                {'attack': 1, 'heart': 1, 'sign': 1}
            ]
            args = parse_args(['-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/index/', '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl', '--predict', '--retrieve', '--output', out_dir, '--cleanOutput','--stats_index_path',  stats_dir])
            actual = get_queries_object(args)
            assert expected[0] == actual.get_query(qid='1')
            assert expected[1] == actual.get_query(qid='2')
            assert expected[2] == actual.get_query(qid='3')

    def test_end_to_end_without_stemmer(self):
        with tempfile.TemporaryDirectory() as out_dir, tempfile.TemporaryDirectory() as stats_dir:
            expected = [
                {'achievements': 1, 'hubble': 1, 'telescope': 1}, 
                {'exit': 1, 'how': 1, 'to': 1, 'vim': 1},
                {'attack': 1, 'heart': 1, 'signs': 1}
            ]
            args = parse_args(['-ti', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-02/index', '--jsonl_queries', os.path.dirname(os.path.realpath(__file__)) + '/resources/small-example-01/queries.jsonl', '--predict', '--retrieve', '--output', out_dir, '--cleanOutput','--stats_index_path',  stats_dir])
            actual = get_queries_object(args)
            assert expected[0] == actual.get_query(qid='1')
            assert expected[1] == actual.get_query(qid='2')
            assert expected[2] == actual.get_query(qid='3')