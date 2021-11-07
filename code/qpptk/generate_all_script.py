#!/usr/bin/env python3.7

import difflib
import subprocess as sp
from glob import glob

INDEX = 'robust04'
# INDEX = 'core17'
# INDEX = 'core18'
CIFF_INDEX_DIR = f'/research/local/olz/ciff_indexes/{INDEX}'
# CIFF_INDEX_DIR = '/research/local/olz/ciff_lite_indexes'
QUERIES_DIR = '/research/local/olz/ciff_query_indexes'
# QUERIES_DIR = '/research/local/olz/ciff_lite_query_indexes'
if INDEX.startswith('core'):
    FILTER_QUERIES = f'/research/local/olz/data/{INDEX}-uqv.qry'
else:
    FILTER_QUERIES = None

query_files = glob(f'{QUERIES_DIR}/*.ciff')
index_files = glob(f'{CIFF_INDEX_DIR}/*.ciff')

python = '/research/local/olz/miniconda3/bin/python3'

for index_file in index_files:
    _query_file = QUERIES_DIR + '/' + index_file.rsplit('/', 1)[1].replace(INDEX, 'robust04').replace('Lucene',
                                                                                                      'Lucene_query')
    if query_files.count(_query_file) == 1:
        query_file = _query_file
    else:
        print(f'\n!!!!!-------!!!!!\n   Query File wasnt found!! {_query_file} \n!!!!!-------!!!!!\n')
        continue
    # query_file = difflib.get_close_matches(, query_files, n = 1, cutoff = 0.8)[0]
    print(f"\nRunning for {index_file}\n")
    sp.run(
        f'PYTHONPATH=./ python3 qpptk/qpptk_main.py --eval -ci {index_file} -cq {query_file} -fq {FILTER_QUERIES}',
        shell=True)
