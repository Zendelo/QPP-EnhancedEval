import argparse
import os
import subprocess as sp
import sys
from glob import glob

import pandas as pd
from syct import timer

from qpptk import Config, set_index_dump_paths, ensure_file, ensure_dir, parse_index_to_db, read_trec_res_file, \
    QueryParserCiff, QueryParserText
from qpptk.global_manager import initialize_text_queries, initialize_ciff_queries, pre_ret_prediction_full, \
    retrieval_full, initialize_db_index, initialize_text_index, initialize_ciff_index, post_ret_prediction_full

parser = argparse.ArgumentParser(description='Run QL retrieval or Query Performance Prediction')
index_group = parser.add_mutually_exclusive_group()
index_group.add_argument('-ti', '--text_index', metavar='INDEX', type=str, default=None, help='path to text index dir')
index_group.add_argument('-ci', '--ciff_index', metavar='INDEX', type=str, default=None, help='path to ciff index file')

queries_group = parser.add_mutually_exclusive_group()
queries_group.add_argument('-tq', '--text_queries', metavar='QUERIES', default=None, help='path to text queries file')
queries_group.add_argument('-cq', '--ciff_queries', metavar='QUERIES', default=None, help='path to ciff queries file')
parser.add_argument('-fq', '--filter_queries', metavar='FilterQUERIES', default=None,
                    help='path to text queries file that will be used to filter the queries')

parser.add_argument('--retrieve', action='store_true', help='add this flag to run retrieval')
parser.add_argument('--method', choices=['ql', 'rm', 'rm_rerank'], default='ql',
                    help='the method to be used in retrieval')
parser.add_argument('--predict', action='store_true', help='add this flag to run predictions using ALL PREDICTORS')
parser.add_argument('--evaluate', action='store_true', help='add this flag to run evaluation')

parser.add_argument('--predPre', action='store_true', help='add this flag to run only pre-retrieval predictors')
parser.add_argument('--predPost', action='store_true', help='add this flag to run only post-retrieval predictors')
# parser.add_argument('--results_file', help='add this flag to run only post-retrieval predictors')

logger = Config.logger

IR_METRICS = {'ap@1000': 'map_cut.1000', 'ap@10': 'map_cut.10', 'ap@100': 'map_cut.100', 'ndcg@10': 'ndcg_cut.10',
              'ndcg@100': 'ndcg_cut.100', 'R-precision': 'Rprec', 'P@1': 'P.1', 'P@5': 'P.5', 'P@10': 'P.10'}

PRECISION = 6


def create_files_in_shell(prefix, eval_metric='ap@1000'):
    """

    :param prefix:
    :param eval_metric:
    """
    corpus, _, stoplist, stemmer = prefix.rsplit('/', 1)[1].split('_')
    metric = IR_METRICS.get(eval_metric, 'map_cut.1000')
    try:
        QL_res_file = ensure_file(f"{prefix}_QL.res")
    except FileNotFoundError as err:
        print(err)
        sys.exit('The results file is not found, add --retrieve to create it')
    trec_eval = Config.TREC_EVAL
    proc = sp.run([trec_eval, '-m', metric, '-qn', f'/research/local/olz/data/{corpus}_mod.qrels', QL_res_file],
                  stderr=sp.STDOUT, stdout=sp.PIPE)
    _fp = open(f"{prefix}_QL.{eval_metric}", 'w')
    sp.run(['awk', '{print $2,$3}'], stdout=_fp, input=proc.stdout)
    _fp.close()
    # sp.run(["sed", "-i", "$ d", f"{prefix}_QL.ap"])  # Remove last line from the file


def load_and_evaluate(prefix, ir_metric, method='pearson', title_only=False):
    # queries = get_queries_object()
    # qids = queries.get_query_ids()
    # from qpptk.utility_functions import duplicate_qrel_file_to_qids
    # corpus, _, stoplist, stemmer = prefix.rsplit('/', 1)[1].split('_')
    # duplicate_qrel_file_to_qids(f'/research/local/olz/data/{corpus}.qrels', qids)
    try:
        eval_df = pd.read_table(f"{prefix}_QL.{ir_metric}", delim_whitespace=True, names=['qid', ir_metric],
                                index_col=0)
    except FileNotFoundError:
        create_files_in_shell(prefix, ir_metric)
        eval_df = pd.read_table(f"{prefix}_QL.{ir_metric}", delim_whitespace=True, names=['qid', ir_metric],
                                index_col=0)
    if title_only:
        eval_df = eval_df.loc[eval_df.index.str.contains('-50-1')]
        title_queries = eval_df.index
        assert len(title_queries) == 249, 'wrong numbers of title queries in ROBUST eval file'
    prediction_files = glob(f"{prefix}*.pre")
    predictors = []
    results = []
    for predictions_file in prediction_files:
        pr_df = pd.read_table(predictions_file, delim_whitespace=True, names=['topic', 'qid', 'prediction'],
                              index_col=[0, 1])
        if title_only:
            pr_df = pr_df.loc[(slice(None), title_queries), :]
            assert len(pr_df) == 249, 'wrong numbers of title queries in ROBUST predictions file'
        print(f"The correlation of {predictions_file}: {pr_df.corrwith(eval_df[ir_metric], method=method)}")
        collection, predictor = predictions_file.rsplit('/', 1)[1].replace('.pre', '').rsplit('_', 1)
        # results[collection] = {predictor: pr_df.corrwith(ap_df[ir_metric])[0]}
        results.append(pr_df.corrwith(eval_df[ir_metric], method=method)[0])
        predictors.append(predictor)
    sr = pd.Series(results, index=predictors)
    sr.name = f'{collection}_{method}'
    return sr


def get_queries_object():
    queries_path, queries_type = (args.text_queries, 'text') if args.text_queries else (args.ciff_queries, 'ciff')
    if queries_path is None:
        queries_path, queries_type = (Config.TEXT_QUERIES, 'text') if Config.TEXT_QUERIES else \
            (Config.CIFF_QUERIES, 'ciff')
        if queries_path is None:
            raise AssertionError('No queries file was specified')
    filter_queries = args.filter_queries
    if filter_queries and filter_queries.lower() != 'none':
        return QueryParserText(queries_path,
                               filter_queries_file=filter_queries) if queries_type == 'text' else QueryParserCiff(
            queries_path, filter_queries_file=filter_queries)
    else:
        return QueryParserText(queries_path) if queries_type == 'text' else QueryParserCiff(queries_path)


def set_index():
    index_path, index_type = (args.text_index, 'text') if args.text_index else (args.ciff_index, 'ciff')
    if index_path is None:
        index_path, index_type = (Config.INDEX_DIR, 'text') if Config.INDEX_DIR else (Config.CIFF_INDEX, 'ciff')
        if index_path is None:
            raise AssertionError('No index was specified')
    return index_path, index_type


@timer
def main():
    results_dir = Config.RESULTS_DIR

    def init_index():
        index = initialize_db_index(db_dir)
        queries = get_queries_object()
        qids = queries.get_query_ids()
        return qids, index, queries

    index_path, index_type = set_index()
    if index_type == 'text':
        dump_files = set_index_dump_paths(index_path)
    prefix = '_'.join(index_path.split('/')[-2:]) if index_type == 'text' else \
        index_path.rsplit('/', 1)[-1].replace('.ciff', '')
    prefix_path = os.path.join(results_dir, prefix)

    if args.retrieve or args.predict or args.predPost or args.predPre:
        _db_dir = os.path.join(Config.DB_DIR, prefix)
        try:
            db_dir = ensure_dir(_db_dir, create_if_not=False)
            qids, index, queries = init_index()
        except FileNotFoundError:
            logger.info(f"Relevant DB index wasn't found in {_db_dir}")
            logger.info(f"Creating a new one")
            index = initialize_text_index(*dump_files) if index_type == 'text' else initialize_ciff_index(index_path)
            parse_index_to_db(index, prefix, Config.DB_DIR)
            db_dir = ensure_dir(_db_dir, create_if_not=False)
            qids, index, queries = init_index()
        if args.retrieve:
            method = args.method
            if method == 'rm_rerank':
                try:
                    results_file = ensure_file(f'{prefix_path}_QL.res')
                except FileNotFoundError:
                    error_msg = f"The file {prefix_path}_QL.res doesn't exist, add --retrieve option to create it first"
                    logger.error(error_msg)
                    sys.exit(error_msg)
                results_df = retrieval_full(qids, index, queries, method=method,
                                            results_df=read_trec_res_file(results_file))
            else:
                results_df = retrieval_full(qids, index, queries, method=method)
            results_df.astype({'qid': str, 'rank': int, 'docScore': float}).to_csv(
                f'{prefix_path}_{method.upper()}.res',
                sep=' ', index=False, header=False,
                float_format=f"%.{PRECISION}f")
        if args.predict:
            args.predPre = True
            args.predPost = True
        if args.predPre:
            predictions_df = pre_ret_prediction_full(qids, index, queries)
            for col in predictions_df.columns:
                predictions_df.loc[:, col].to_csv(f"{prefix_path}_PRE_{col}.pre", sep=' ', index=True, header=False,
                                                  float_format=f"%.{PRECISION}f")
        if args.predPost:
            retrieval_method = 'QL'
            try:
                results_file = ensure_file(f'{prefix_path}_{retrieval_method}.res')
            except FileNotFoundError:
                error_msg = f"The file {prefix_path}_{retrieval_method}.res doesn't exist," \
                            f"add --retrieve option to create it first"
                logger.error(error_msg)
                sys.exit(error_msg)
            predictions_df = post_ret_prediction_full(qids, index, queries, read_trec_res_file(results_file))
            for col in predictions_df.columns:
                predictions_df.loc[:, col].to_csv(f"{prefix_path}_{retrieval_method}_{col}.pre", sep=' ', index=True,
                                                  header=False,
                                                  float_format=f"%.{PRECISION}f")
    if args.evaluate:
        # method = 'spearman'
        method = 'kendall'
        # method = 'pearson'
        # queries = 'title'
        queries = 'all'
        ir_metric = 'ap@1000'
        # ir_metric = 'ndcg@10'
        title_only = True if queries == 'title' else False
        load_and_evaluate(prefix_path, ir_metric, method=method, title_only=title_only).to_pickle(
            f'{prefix_path}.eval_{ir_metric}_{method}_{queries}_queries.pkl')


if __name__ == '__main__':
    args = parser.parse_args()  # defined and used as a "global" variable
    main()
