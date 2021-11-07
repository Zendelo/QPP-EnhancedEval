import multiprocessing as mp

import pandas as pd
from syct import timer

from qpptk import Config, IndexText, IndexDB, QueryParserText, QueryParserCiff, LocalManagerRetrieval, parse_index_file, \
    IndexCiff, add_topic_to_qdf, LocalManagerPredictorPre, LocalManagerPredictorPost

logger = Config.logger


def init_proc(*args):
    global index
    global queries
    global results_df
    index = args[0]
    queries = args[1]
    results_df = args[2] if len(args) > 2 else None


def run_pre_prediction_process(qid):
    process = LocalManagerPredictorPre(index_obj=index, query_obj=queries, qid=qid)
    max_idf = process.calc_max_idf()
    avg_idf = process.calc_avg_idf()
    scq = process.calc_scq()
    max_scq = process.calc_max_scq()
    avg_scq = process.calc_avg_scq()
    var = process.calc_var()
    avg_var = process.calc_avg_var()
    max_var = process.calc_max_var()
    logger.debug(qid + ' finished')
    return {'qid': qid, 'max-idf': max_idf, 'avg-idf': avg_idf, 'scq': scq, 'max-scq': max_scq, 'avg-scq': avg_scq,
            'var': var, 'max-var': max_var, 'avg-var': avg_var}


def run_post_prediction_process(qid):
    process = LocalManagerPredictorPost(index_obj=index, query_obj=queries, results_df=results_df, qid=qid)
    wig = process.calc_wig(Config.WIG_LIST_SIZE)
    nqc = process.calc_nqc(Config.NQC_LIST_SIZE)
    smv = process.calc_smv(Config.SMV_LIST_SIZE)

    ret_process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    results_vec = results_df.loc[qid, ['docNo', 'docScore']].reset_index(drop=True).values
    p_w_rm = ret_process.generate_rm(results_vec[:Config.CLARITY_LIST_SIZE])
    clarity_list_size = min(Config.CLARITY_FB_TERMS, len(p_w_rm['term_id']))
    if clarity_list_size < Config.CLARITY_FB_TERMS:
        logger.warn(f'Query-{qid}: The RM passed to Clarity had less terms than clarity_list_size parameter')
    clarity = process.calc_clarity(p_w_rm=p_w_rm[:clarity_list_size])

    re_ranked_df = run_rm_rerank_retrieval_process(qid)

    uef_wig = process.calc_uef(Config.WORKING_SET_SIZE, re_ranked_df.set_index('qid'), wig)
    uef_nqc = process.calc_uef(Config.WORKING_SET_SIZE, re_ranked_df.set_index('qid'), nqc)
    uef_smv = process.calc_uef(Config.WORKING_SET_SIZE, re_ranked_df.set_index('qid'), smv)
    uef_clarity = process.calc_uef(Config.WORKING_SET_SIZE, re_ranked_df.set_index('qid'), clarity)

    logger.debug(qid + ' finished')
    return {'qid': qid, 'wig': wig, 'nqc': nqc, 'smv': smv, 'clarity': clarity, 'uef-wig': uef_wig, 'uef-nqc': uef_nqc,
            'uef-smv': uef_smv, 'uef-clarity': uef_clarity}


def _run_multiprocess_sync(func, tasks, n_proc, *init_args):
    with mp.Pool(processes=n_proc, initializer=init_proc, initargs=init_args) as pool:
        result = pool.map(func, tasks)
    return result


def pre_ret_prediction_full(qids, index, queries, n_proc=Config.N_PROC):
    # result = {}
    # for qid in qids:
    #     result[qid] = run_prediction_process(qid, index, queries)
    # df = pd.DataFrame.from_dict(result, orient='index')
    # df.index = df.index.rename('qid')
    result = _run_multiprocess_sync(run_pre_prediction_process, qids, n_proc, index, queries)
    df = pd.DataFrame(result)
    logger.debug(df)
    return add_topic_to_qdf(df).set_index(['topic', 'qid'])


def post_ret_prediction_full(qids, index, queries, results_df, n_proc=Config.N_PROC):
    result = _run_multiprocess_sync(run_post_prediction_process, qids, n_proc, index, queries, results_df)
    df = pd.DataFrame(result)
    logger.debug(df)
    return add_topic_to_qdf(df).set_index(['topic', 'qid'])


@timer
def run_ql_retrieval_process(qid):
    columns = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']
    process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    result = process.run_ql_retrieval()
    logger.debug(qid + ' finished')
    df = pd.DataFrame.from_records(result, columns=['docNo', 'docScore'])
    return df.assign(qid=qid, iteration='Q0', rank=range(1, len(result) + 1), method='QL')[columns]


@timer
def run_rm_retrieval_process(qid):
    columns = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']
    process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    result = process.run_rm_retrieval()
    logger.debug(qid + ' finished')
    df = pd.DataFrame.from_records(result, columns=['docNo', 'docScore'])
    return df.assign(qid=qid, iteration='Q0', rank=range(1, len(result) + 1), method='RM')[columns]


@timer
def run_rm_rerank_retrieval_process(qid, return_rm=False):
    columns = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']
    process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    results_vec = None if results_df is None else results_df.loc[qid, ['docNo', 'docScore']].reset_index(
        drop=True).values
    if Config.WORKING_SET_SIZE:
        results_vec = results_vec[:Config.WORKING_SET_SIZE]
    result, p_w_rm = process.run_rm_retrieval(re_rank_ql=True, working_set_docs=results_vec)
    logger.debug(qid + ' finished')
    df = pd.DataFrame.from_records(result, columns=['docNo', 'docScore'])
    res_df = df.assign(qid=qid, iteration='Q0', rank=range(1, len(result) + 1), method='QlRm')[columns]
    if return_rm:
        return res_df, p_w_rm
    else:
        return res_df


def retrieval_full(qids, index, queries, n_proc=Config.N_PROC, method='ql', results_df=None):
    # init_proc(index, queries)
    # for qid in qids:
    #     run_retrieval_process(qid)
    if method == 'ql':
        result = _run_multiprocess_sync(run_ql_retrieval_process, qids, n_proc, index, queries)
    elif method == 'rm':
        result = _run_multiprocess_sync(run_rm_retrieval_process, qids, n_proc, index, queries)
    elif method == 'rm_rerank':
        result = _run_multiprocess_sync(run_rm_rerank_retrieval_process, qids, n_proc, index, queries, results_df)
    else:
        logger.warn(f'Unknown method, choose between [ql, rm, rm_rerank]')
        return None
    df = pd.concat(result)
    logger.debug(df)
    return df


@timer
def initialize_ciff_index(ciff_index):
    header, terms_dict, doc_records = parse_index_file(ciff_index)
    return IndexCiff(header, ciff_index, terms_dict, doc_records)


@timer
def initialize_text_index(text_inv, dict_txt, doc_lens, doc_names, index_globals):
    return IndexText(text_inverted=text_inv, terms_dict=dict_txt, index_global=index_globals, document_lengths=doc_lens,
                     document_names=doc_names)


@timer
def initialize_db_index(db_dir):
    return IndexDB(index_db_dir=db_dir)


@timer
def initialize_ciff_queries(queries_file):
    return QueryParserCiff(queries_file)


@timer
def initialize_text_queries(queries_file):
    return QueryParserText(queries_file)

# @timer
# def main():
#     index = initialize_ciff_index(CIFF_INDEX)
#     # index = initialize_text_index(*Config.set_dump_paths(INDEX_DIR))
#     # queries = initialize_ciff_queries(QUERIES_FILE)
#     queries = initialize_text_queries(QUERIES_FILE)
#     qids = queries.get_query_ids()
#     # result = [run_retrieval_process(qids[1], index, queries)]
#     # retrieval_full(qids, index, queries)
#     prediction_full(qids, index, queries)
#
#
# if __name__ == '__main__':
#     main()
#     # test()
