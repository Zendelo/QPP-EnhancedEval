import argparse
import os
import shutil
import subprocess as sp
import sys
from pathlib import Path
from glob import glob

import pandas as pd
from sklearn.metrics import pairwise_distances
from syct import timer

from qpptk import Config, set_index_dump_paths, ensure_file, ensure_dir, read_trec_res_file, \
    QueryParserCiff, QueryParserText, QueryParserJsonl, add_topic_to_qdf, calc_ndcg
from qpptk.global_manager import pre_ret_prediction_full, \
    retrieval_full, initialize_db_index, initialize_text_index, post_ret_prediction_full, \
    initialize_terrier_index

def parse_args(args):
    parser = argparse.ArgumentParser(description='Run QL retrieval or Query Performance Prediction')
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument('--text_index', metavar='INDEX', type=str, default=None, help='path to text index dir')
    index_group.add_argument('-ci', '--ciff_index', metavar='INDEX', type=str, default=None, help='path to ciff index file')
    index_group.add_argument('-ti', '--terrier_index', metavar='INDEX', type=str, default=None,
                         help='path to terrier index dir')
    

    queries_group = parser.add_mutually_exclusive_group()
    queries_group.add_argument('-tq', '--text_queries', metavar='QUERIES', default=None, help='path to text queries file')
    queries_group.add_argument('-cq', '--ciff_queries', metavar='QUERIES', default=None, help='path to ciff queries file')
    queries_group.add_argument('-jq', '--jsonl_queries', metavar='QUERIES', default=None, help='path to jsonl queries file')

    parser.add_argument('-fq', '--filter_queries', metavar='FilterQUERIES', default=None,
                        help='path to text queries file that will be used to filter the queries')
    parser.add_argument('-dq', '--drop_queries', metavar='DropQUERIES', default=None,
                        help='path to text query ids file that will be used to drop the queries')

    parser.add_argument('--retrieve', action='store_true', help='add this flag to run retrieval')
    parser.add_argument('--method', choices=['ql', 'rm', 'rm_rerank'], default='ql',
                        help='the method to be used in retrieval')
    parser.add_argument('--predict', action='store_true', help='add this flag to run predictions using ALL PREDICTORS')
    parser.add_argument('--evaluate', action='store_true', help='add this flag to run evaluation')
    parser.add_argument('--pairs_sim', action='store_true', help='Generates a file with similarity between all query pairs')
    parser.add_argument('--cluster_queries', action='store_true',
                        help='Splits the queries file into separate files by similarity')
    parser.add_argument('--keep_duplicate_queries', action='store_true',
                        help='By default duplicate queries per topic will be removed, use this option to keep them')

    parser.add_argument('--predPre', action='store_true', help='add this flag to run only pre-retrieval predictors')
    parser.add_argument('--predPost', action='store_true', help='add this flag to run only post-retrieval predictors')

    parser.add_argument('--output', default=None, required=False, help='The output directory')
    parser.add_argument('--cleanOutput', action='store_true', help='Clean all temporary output files and output only a joined jsonl file')
    parser.add_argument('--stats_index_path', type=str, default=None, help='location of the index statistics')
    parser.add_argument('--run-file', type=Path, default=None, help='path to run file to be used for prediction post retrieval predictions')

    return parser.parse_args(args)

logger = Config.logger

TREC_IR_METRICS = {'ap@1000': 'map_cut.1000', 'ap@10': 'map_cut.10', 'ap@100': 'map_cut.100', 'ndcg@10': 'ndcg_cut.10',
                   'ndcg@100': 'ndcg_cut.100', 'R-precision': 'Rprec', 'P@1': 'P.1', 'P@5': 'P.5', 'P@10': 'P.10',
                   'RR': 'recip_rank'}

PRECISION = 6


def generate_trec_eval(prefix, eval_metric='ap@1000'):
    """

    :param prefix:
    :param eval_metric:
    """
    # corpus, _, stoplist, stemmer = prefix.rsplit('/', 1)[1].split('_')
    metric = TREC_IR_METRICS.get(eval_metric, 'map_cut.1000')
    try:
        ql_res_file = ensure_file(f"{prefix}_QL.res")
    except FileNotFoundError as err:
        print(err)
        sys.exit('The results file is not found, add --retrieve to create it')
    trec_eval = Config.TREC_EVAL
    qrels_file = ensure_file(Config.QREL_FILE)
    proc = sp.run([trec_eval, '-m', metric, '-qn', qrels_file, ql_res_file],
                  stderr=sp.STDOUT, stdout=sp.PIPE)
    _fp = open(f"{prefix}_QL.{eval_metric}", 'w')
    sp.run(['awk', '{print $2,$3}'], stdout=_fp, input=proc.stdout)
    _fp.close()
    # sp.run(["sed", "-i", "$ d", f"{prefix}_QL.ap"])  # Remove last line from the file


def generate_rbp_eval(prefix, eval_metric='rbp-0.95'):
    """

    :param eval_metric:
    :param prefix:
    """
    # corpus, _, stoplist, stemmer = prefix.rsplit('/', 1)[1].split('_')
    try:
        ql_res_file = ensure_file(f"{prefix}_QL.res")
    except FileNotFoundError as err:
        print(err)
        sys.exit('The results file is not found, add --retrieve to create it')
    metric, p = eval_metric.split('-')
    rbp_eval = Config.RBP_EVAL
    qrels_file = ensure_file(Config.QREL_FILE)
    proc = sp.run([rbp_eval, '-q', '-T', '-H', '-W', '-p', p, qrels_file, ql_res_file],
                  stderr=sp.STDOUT, stdout=sp.PIPE)
    _fp = open(f"{prefix}_QL.{eval_metric}", 'w')
    sp.run(['awk', '{print $4,$8}'], stdout=_fp, input=proc.stdout)
    _fp.close()
    # sp.run(["sed", "-i", "$ d", f"{prefix}_QL.ap"])  # Remove last line from the file


@timer
def create_ndcg_eval_files(prefix, ir_metric):
    try:
        ql_res_file = ensure_file(f"{prefix}_QL.res")
        qrels_file = ensure_file(Config.QREL_FILE)
    except FileNotFoundError as err:
        print(err)
        sys.exit('The results file or qrels file is not found, adding --retrieve will generate a results file')
    k = int(ir_metric.rsplit('@', 1)[-1])
    calc_ndcg(qrels_file, ql_res_file, k, original=False, logger=logger)


def load_and_evaluate(prefix, ir_metric, method='pearson', title_only=False):
    queries = get_queries_object()
    qids = queries.get_query_ids()
    _components = prefix.rsplit('/', 1)[1].split('_')
    if len(_components) > 3:
        corpus, _, stoplist, stemmer = _components
    elif len(_components) == 3:
        corpus, stoplist, stemmer = _components
    else:
        print('Unknown format of index name')
    try:
        eval_df = pd.read_table(f"{prefix}_QL.{ir_metric}", delim_whitespace=True, names=['qid', ir_metric],
                                index_col=0)
    except FileNotFoundError:
        if ir_metric.startswith('ndcg'):
            create_ndcg_eval_files(prefix, ir_metric)
        elif ir_metric.startswith('rbp'):
            generate_rbp_eval(prefix, ir_metric)
        else:
            generate_trec_eval(prefix, ir_metric)
        eval_df = pd.read_table(f"{prefix}_QL.{ir_metric}", delim_whitespace=True, names=['qid', ir_metric],
                                index_col=0)
    if title_only:
        eval_df = eval_df.loc[eval_df.index.str.contains('-50-1')]
        title_queries = eval_df.index
        # assert len(title_queries) == 249, 'wrong numbers of title queries in ROBUST eval file'
    prediction_files = glob(f"{prefix}*.pre")
    predictors = []
    results = []
    for predictions_file in prediction_files:
        pr_df = pd.read_table(predictions_file, delim_whitespace=True, names=['topic', 'qid', 'prediction'],
                              index_col=[0, 1])
        if title_only:
            pr_df = pr_df.loc[(slice(None), title_queries), :]
            # assert len(pr_df) == 249, 'wrong numbers of title queries in ROBUST predictions file'
        # print(f"The correlation of {predictions_file}: {pr_df.corrwith(eval_df[ir_metric], method=method)}")
        collection, predictor = predictions_file.rsplit('/', 1)[1].replace('.pre', '').rsplit('_', 1)
        # results[collection] = {predictor: pr_df.corrwith(ap_df[ir_metric])[0]}
        results.append(pr_df.corrwith(eval_df[ir_metric], method=method)[0])
        predictors.append(predictor)
    sr = pd.Series(results, index=predictors)
    sr.name = f'{collection}_{method}_{ir_metric}'
    print(sr.sort_index())
    print(sr.sort_values())
    return sr


def get_queries_object(args):
    drop_duplicates = not args.keep_duplicate_queries
    if args.text_queries:
        queries_path, queries_type = args.text_queries, 'text'
    elif args.ciff_queries:
        queries_path, queries_type = args.ciff_queries, 'ciff'
    elif args.jsonl_queries:
        queries_path, queries_type = args.jsonl_queries, 'jsonl'
    else:
        queries_path, queries_type = None, None

    if queries_path is None:
        if Config.TEXT_QUERIES:
            queries_path, queries_type = Config.TEXT_QUERIES, 'text'
        elif Config.CIFF_QUERIES:
            queries_path, queries_type = Config.CIFF_QUERIES, 'ciff'
        elif Config.JSONL_QUERIES:
            queries_path, queries_type = Config.JSONL_QUERIES, 'jsonl'
        else:
            raise AssertionError('No queries file was specified')
    filter_queries = args.filter_queries
    if filter_queries and filter_queries.lower() != 'none':
        return QueryParserText(queries_path,
                               filter_queries_file=filter_queries,
                               drop_duplicate_queries=drop_duplicates) if queries_type == 'text' else QueryParserCiff(
            queries_path, filter_queries_file=filter_queries, drop_duplicate_queries=drop_duplicates)
    drop_queries = args.drop_queries
    if drop_queries and drop_queries.lower() != 'none':
        if queries_type == 'text':
            return QueryParserText(queries_path, drop_queries_file='duplicated_qids.txt',
                                   drop_duplicate_queries=drop_duplicates)
        elif queries_type == 'ciff':
            QueryParserCiff(queries_path, drop_queries_file='duplicated_qids.txt',
                            drop_duplicate_queries=drop_duplicates)
        elif queries_type == 'jsonl':
            QueryParserJsonl(queries_path, args.terrier_index, drop_queries_file='duplicated_qids.txt',
                             drop_duplicate_queries=drop_duplicates)
    else:
        if queries_type == 'text':
            return QueryParserText(queries_path, drop_duplicate_queries=drop_duplicates)
        elif queries_type == 'ciff':
            return QueryParserCiff(queries_path, drop_duplicate_queries=drop_duplicates)
        elif queries_type == 'jsonl':
            return QueryParserJsonl(queries_path, args.terrier_index, drop_duplicate_queries=drop_duplicates)


def set_index_paths(args):
    index_path, index_type = (args.ciff_index, 'ciff') if args.ciff_index else (
        (args.text_index, 'text') if args.text_index else (args.terrier_index, 'terrier'))
    if index_path is None:
        if Config.CIFF_INDEX:
            index_path, index_type = Config.CIFF_INDEX, 'ciff'
        elif Config.TERRIER_INDEX:
            index_path, index_type = Config.TERRIER_INDEX, 'terrier'
        else:
            index_path, index_type = Config.INDEX_DIR, 'text'
        if index_path is None:
            raise AssertionError('No index was specified')
    return index_path, index_type


def generate_pairs_similarity(queries):
    try:
        df = queries.get_queries_df()
    except AttributeError:
        df = add_topic_to_qdf(
            pd.DataFrame.from_dict(queries.queries_dict, orient='index').rename_axis('qid')).set_index('topic').drop(
            '672', errors='ignore').set_index('qid')
    df = df.sort_index()
    cos_sim_df = pd.DataFrame(1 - pairwise_distances(df.fillna(0).to_numpy(), metric='cosine', n_jobs=10),
                              index=df.index, columns=df.index)
    jac_df = pd.DataFrame(1 - pairwise_distances(df.fillna(0).astype(bool).to_numpy(), metric='jaccard', n_jobs=10),
                          index=df.index, columns=df.index)
    dsc_df = pd.DataFrame(1 - pairwise_distances(df.fillna(0).astype(bool).to_numpy(), metric='dice', n_jobs=10),
                          index=df.index, columns=df.index)
    return cos_sim_df, jac_df, dsc_df


def cluster_queries(queries):
    queries.cluster_queries_by_similarity(3)


@timer
def main(args):
    if args.output:
        Config.RESULTS_DIR = args.output
    results_dir = Config.RESULTS_DIR

    def init_db_index():
        index = initialize_db_index(db_dir)
        queries = get_queries_object(args)
        qids = queries.get_query_ids()
        return qids, index, queries

    def init_readonly_index():  # TODO: should be init for terrier index
        queries = get_queries_object(args)
        qids = queries.get_query_ids()
        index = initialize_terrier_index(index_path, partial_terms=queries.get_queries_df().columns, stats_index_path=args.stats_index_path)
        index_hash = index().partial_terms_hash
        return qids, index, queries, index_hash

    def init_writeable_index():
        queries = get_queries_object(args)
        return initialize_terrier_index(index_path, partial_terms=queries.get_queries_df().columns, read_only=False, stats_index_path=args.stats_index_path)()

    index_path, index_type = set_index_paths(args)
    if index_type == 'text':
        dump_files = set_index_dump_paths(index_path)
    if index_type == 'ciff':
        prefix = index_path.rsplit('/', 1)[-1].replace('.ciff', '')
    elif index_type == 'terrier':
        prefix = index_path.rsplit('/', 1)[-1]
    else:
        prefix = '_'.join(index_path.split('/')[-2:])
    prefix_path = os.path.join(results_dir, prefix)

    if args.retrieve or args.predict or args.predPost or args.predPre:
        if index_type == 'ciff' or index_type == 'text':
            _db_dir = os.path.join(Config.DB_DIR, prefix)
            try:
                db_dir = ensure_dir(_db_dir, create_if_not=False)
                qids, index, queries = init_db_index()
            except FileNotFoundError:
                logger.info(f"Relevant DB index wasn't found in {_db_dir}")
                logger.info(f"Creating a new one")
                if index_type == 'ciff':
                    index = initialize_ciff_index(index_path)
                    parse_index_to_db(index, prefix, Config.DB_DIR)
                    db_dir = ensure_dir(_db_dir, create_if_not=False)
                else:
                    index = initialize_text_index(*dump_files)
                    parse_index_to_db(index, prefix, Config.DB_DIR)
                    db_dir = ensure_dir(_db_dir, create_if_not=False)
                qids, index, queries = init_db_index()
        elif index_type == 'terrier':
            try:
                qids, index, queries, index_hash = init_readonly_index()
            except FileNotFoundError:
                init_writeable_index()
                qids, index, queries, index_hash = init_readonly_index()
        else:
            sys.exit('No known index type was specified')
        try:
            path, collection = prefix_path.rsplit('/', 1)
            prefix_path = os.path.join(ensure_dir(os.path.join(path, str(index_hash))), prefix)
        except NameError:
            pass

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

        combined_predictions = []
        if args.predPre:
            predictions_df = pre_ret_prediction_full(qids, index, queries)
            combined_predictions += [predictions_df]
            for col in predictions_df.columns:
                predictions_df.loc[:, col].to_csv(f"{prefix_path}_PRE_{col}.pre", sep=' ', index=True,
                                                  header=False,
                                                  float_format=f"%.{PRECISION}f")
        if args.predPost:
            retrieval_method = 'QL'
            try:
                results_file = ensure_file(f'{prefix_path}_{retrieval_method}.res' if args.run_file is None else args.run_file)
            except FileNotFoundError:
                error_msg = f"The file {prefix_path}_{retrieval_method}.res doesn't exist," \
                            f"add --retrieve option to create it first"
                logger.error(error_msg)
                sys.exit(error_msg)
            predictions_df = post_ret_prediction_full(qids, index, queries, read_trec_res_file(results_file))
            combined_predictions += [predictions_df]
            for col in predictions_df.columns:
                predictions_df.loc[:, col].to_csv(f"{prefix_path}_{retrieval_method}_{col}.pre", sep=' ',
                                                  index=True, header=False, float_format=f"%.{PRECISION}f")
        
        if combined_predictions:
            qid_to_preds = {}
            for df in combined_predictions:
                for _, i in df.iterrows():
                    if i['qid'] not in qid_to_preds:
                        qid_to_preds[i['qid']] = {}
                    for k,v in i.items():
                        if k == 'topic':
                            continue
                        qid_to_preds[i['qid']][k] = v

            pd.DataFrame([v for _, v in qid_to_preds.items()]).to_json(results_dir + '/queries.jsonl', lines=True, orient='records')

    if args.cleanOutput:
        for d in os.listdir(args.output):
            if os.path.exists(args.output + '/' + d) and os.path.isdir(args.output + '/' + d):
                shutil.rmtree(args.output + '/' + d)

    if args.evaluate:
        method = 'pearson'
        queries = 'all'
        ir_metric = 'ap@1000'

        title_only = True if queries == 'title' else False
        if Config.BATCH_NAME:
            _prefix = prefix_path[::-1].replace('/', ' ', 1)[::-1].replace(' ', '/' + Config.BATCH_NAME + '/')
        else:
            _prefix = prefix_path
        load_and_evaluate(_prefix, ir_metric, method=method, title_only=title_only).to_pickle(
            f'{prefix_path}.eval_{ir_metric}_{method}_{queries}_queries.pkl')

    if args.pairs_sim:
        cos_sim_df, jac_df, dsc_df = generate_pairs_similarity(get_queries_object(args))
        cos_sim_df.to_csv(f"{prefix_path}_pairwise_sim-cos.tsv", sep=' ', index=True, header=True,
                          float_format=f"%.{PRECISION}f")
        jac_df.to_csv(f"{prefix_path}_pairwise_sim-jac.tsv", sep=' ', index=True, header=True,
                      float_format=f"%.{PRECISION}f")
        dsc_df.to_csv(f"{prefix_path}_pairwise_sim-dsc.tsv", sep=' ', index=True, header=True,
                      float_format=f"%.{PRECISION}f")

    if args.cluster_queries:
        cluster_queries(get_queries_object(args))


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
