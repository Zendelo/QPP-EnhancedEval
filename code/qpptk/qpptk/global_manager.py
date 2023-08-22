import multiprocessing as mp
from functools import partial

import pandas as pd
import syct
from syct import timer

from qpptk import Config, IndexText, IndexDB, QueryParserText, QueryParserCiff, LocalManagerRetrieval, \
    add_topic_to_qdf, LocalManagerPredictorPre, LocalManagerPredictorPost, IndexTerrier, \
    ensure_dir

logger = Config.logger
TREC_RES_COLUMNS = ['qid', 'iteration', 'docNo', 'rank', 'docScore', 'method']


def init_proc(*args):
    kwargs = dict(args)
    global index
    global queries
    global results_df
    if callable(kwargs.get('index')):
        index = kwargs.get('index')()
    else:
        index = kwargs.get('index')
    queries = kwargs.get('queries')
    results_df = kwargs.get('results_df')
    ensure_dir(f'{Config.RESULTS_DIR}/temp/{index}', create_if_not=True)


def run_pre_prediction_process(qid):
    timer = syct.Timer(f'QID: {qid}', logger=logger)
    process = LocalManagerPredictorPre(index_obj=index, query_obj=queries, qid=qid)
    max_idf = process.calc_max_idf()
    avg_idf = process.calc_avg_idf()
    scq = process.calc_scq()
    max_scq = process.calc_max_scq()
    avg_scq = process.calc_avg_scq()
    var = process.calc_var()
    avg_var = process.calc_avg_var()
    max_var = process.calc_max_var()
    # logger.debug(qid + ' finished')
    timer.stop()
    return {'qid': qid, 'max-idf': max_idf, 'avg-idf': avg_idf, 'scq': scq, 'max-scq': max_scq, 'avg-scq': avg_scq,
            'var': var, 'max-var': max_var, 'avg-var': avg_var}


def run_post_prediction_process(qid):
    timer = syct.Timer(f'QID: {qid}', logger=logger)
    process = LocalManagerPredictorPost(index_obj=index, query_obj=queries, results_df=results_df, qid=qid)
    wig = process.calc_wig(Config.WIG_LIST_SIZE)
    nqc = process.calc_nqc(Config.NQC_LIST_SIZE)
    smv = process.calc_smv(Config.SMV_LIST_SIZE)

    ret_process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    results_vec = results_df.loc[qid, ['docNo', 'docScore']].reset_index(drop=True).values

    c_p_w_rm = ret_process.generate_rm(results_vec[:Config.CLARITY_LIST_SIZE])
    # if Config.QF_LIST_SIZE == Config.CLARITY_LIST_SIZE:
    #     q_p_w_rm = c_p_w_rm
    # else:
    #     q_p_w_rm = ret_process.generate_rm(results_vec[:Config.QF_LIST_SIZE])
    # if Config.UEF_LIST_SIZE == Config.CLARITY_LIST_SIZE:
    #     u_p_w_rm = c_p_w_rm
    # elif Config.UEF_LIST_SIZE == Config.QF_LIST_SIZE:
    #     u_p_w_rm = q_p_w_rm
    # else:
    #     u_p_w_rm = ret_process.generate_rm(results_vec[:Config.UEF_LIST_SIZE])

    clarity_terms_size = min(Config.CLARITY_FB_TERMS, len(c_p_w_rm['term_id']))
    if clarity_terms_size < Config.CLARITY_FB_TERMS:
        logger.warn(f'Query-{qid}: The RM passed to Clarity had less terms than clarity_list_size parameter')
    clarity = process.calc_clarity(p_w_rm=c_p_w_rm[:clarity_terms_size])

    # rm_ranked_df, _ = ret_process.run_rm_retrieval(initial_set_docs=results_vec,
    #                                                _sorted_rm_terms=q_p_w_rm[:Config.QF_FB_TERMS])
    # _df = pd.DataFrame.from_records(rm_ranked_df, columns=['docNo', 'docScore'])
    # rm_df = _df.assign(qid=qid, iteration='Q0', rank=range(1, len(rm_ranked_df) + 1), method='RM')[TREC_RES_COLUMNS]
    # qf = process.calc_qf(Config.QF_OVERLAP_SIZE, rm_df)
    #
    # rm_re_ranked_df, _ = ret_process.run_rm_retrieval(ranking_set_docs=results_vec[:Config.UEF_RANKING_SIZE],
    #                                                   initial_set_docs=results_vec,
    #                                                   _sorted_rm_terms=u_p_w_rm[:Config.UEF_FB_TERMS])
    # _df = pd.DataFrame.from_records(rm_re_ranked_df, columns=['docNo', 'docScore'])
    # re_ranked_df = _df.assign(qid=qid, iteration='Q0', rank=range(1, len(rm_re_ranked_df) + 1),
    #                           method='RM')[TREC_RES_COLUMNS]
    # uef_wig = process.calc_uef(Config.UEF_RANKING_SIZE, re_ranked_df.set_index('qid'), wig)
    # uef_nqc = process.calc_uef(Config.UEF_RANKING_SIZE, re_ranked_df.set_index('qid'), nqc)
    # uef_smv = process.calc_uef(Config.UEF_RANKING_SIZE, re_ranked_df.set_index('qid'), smv)
    # uef_clarity = process.calc_uef(Config.UEF_RANKING_SIZE, re_ranked_df.set_index('qid'), clarity)
    # uef_qf = process.calc_uef(Config.UEF_RANKING_SIZE, re_ranked_df.set_index('qid'), qf)

    timer.stop()
    # return {'qid': qid, f'wig+{Config.WIG_LIST_SIZE}': wig, f'nqc+{Config.NQC_LIST_SIZE}': nqc,
    #         f'smv+{Config.SMV_LIST_SIZE}': smv, f'clarity+{Config.CLARITY_LIST_SIZE}+{clarity_terms_size}': clarity,
    #         f'qf+{Config.QF_LIST_SIZE}+{Config.QF_OVERLAP_SIZE}': qf,
    #         f'uef+{Config.UEF_LIST_SIZE}+{Config.UEF_RANKING_SIZE}-wig+{Config.WIG_LIST_SIZE}': uef_wig,
    #         f'uef+{Config.UEF_LIST_SIZE}+{Config.UEF_RANKING_SIZE}-nqc+{Config.NQC_LIST_SIZE}': uef_nqc,
    #         f'uef+{Config.UEF_LIST_SIZE}+{Config.UEF_RANKING_SIZE}-smv+{Config.SMV_LIST_SIZE}': uef_smv,
    #         f'uef+{Config.UEF_LIST_SIZE}+{Config.UEF_RANKING_SIZE}-clarity+{Config.CLARITY_LIST_SIZE}+{clarity_terms_size}': uef_clarity,
    #         f'uef+{Config.UEF_LIST_SIZE}+{Config.UEF_RANKING_SIZE}-qf+{Config.QF_LIST_SIZE}+{Config.QF_FB_TERMS}+{Config.QF_OVERLAP_SIZE}': uef_qf}
    return {'qid': qid, f'wig+{Config.WIG_LIST_SIZE}': wig, f'nqc+{Config.NQC_LIST_SIZE}': nqc,
            f'smv+{Config.SMV_LIST_SIZE}': smv, f'clarity+{Config.CLARITY_LIST_SIZE}+{clarity_terms_size}': clarity}


def _run_multiprocess_sync(func, tasks, n_proc, **init_kwargs):
    with mp.Pool(processes=n_proc, initializer=init_proc, initargs=init_kwargs.items()) as pool:
        # result = pool.map(func, tasks, chunksize=len(tasks) // n_proc)
        result = pool.map(func, tasks)
    return result


# @timer(info)
def pre_ret_prediction_full(qids, index, queries, n_proc=Config.N_PROC):
    result = _run_multiprocess_sync(run_pre_prediction_process, qids, n_proc, index=index, queries=queries)
    df = pd.DataFrame(result)
    logger.debug(df)
    return add_topic_to_qdf(df).set_index(['topic', 'qid'])


# @timer(info)
def post_ret_prediction_full(qids, index, queries, results_df, n_proc=Config.N_PROC):
    result = _run_multiprocess_sync(run_post_prediction_process, qids, n_proc, index=index, queries=queries,
                                    results_df=results_df)
    df = pd.DataFrame(result)
    logger.debug(df)
    return add_topic_to_qdf(df).set_index(['topic', 'qid'])


# @timer  # TODO: add debug level, and should add debugging each qid with time to a log file
def run_ql_retrieval_process(qid):
    timer = syct.Timer(f'QID: {qid}', logger=logger)
    process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    result = process.run_ql_retrieval()
    timer.stop()
    # logger.debug(qid + ' finished')
    df = pd.DataFrame.from_records(result, columns=['docNo', 'docScore'])
    result_df = df.assign(qid=qid, iteration='Q0', rank=range(1, len(result) + 1), method='QL')[TREC_RES_COLUMNS]
    result_df.to_pickle(f'{Config.RESULTS_DIR}/temp/{index}/{qid}_QL.res.pkl')
    return result_df


@timer  # TODO: add debug level
def run_rm_retrieval_process(qid, return_rm=False, __sorted_rm_terms=None):
    timer = syct.Timer(f'QID: {qid}', logger=logger)
    process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    result, p_w_rm = process.run_rm_retrieval()
    logger.debug(qid + ' finished')
    df = pd.DataFrame.from_records(result, columns=['docNo', 'docScore'])
    res_df = df.assign(qid=qid, iteration='Q0', rank=range(1, len(result) + 1), method='RM')[TREC_RES_COLUMNS]
    timer.stop()
    if return_rm:
        return res_df, p_w_rm
    else:
        return res_df


@timer  # TODO: add debug level
def run_rm_rerank_retrieval_process(qid, ranking_set_docs=None, return_rm=False, __sorted_rm_terms=None):
    timer = syct.Timer(f'QID: {qid}', logger=logger)
    process = LocalManagerRetrieval(index_obj=index, query_obj=queries, qid=qid)
    results_vec = ranking_set_docs if ranking_set_docs is not None else results_df.loc[
        qid, ['docNo', 'docScore']].reset_index(drop=True).values
    if Config.WORKING_SET_SIZE:
        results_vec = results_vec[:Config.WORKING_SET_SIZE]
    result, p_w_rm = process.run_rm_retrieval(ranking_set_docs=results_vec)
    logger.debug(qid + ' finished')
    df = pd.DataFrame.from_records(result, columns=['docNo', 'docScore'])
    res_df = df.assign(qid=qid, iteration='Q0', rank=range(1, len(result) + 1), method='QlRm')[TREC_RES_COLUMNS]
    timer.stop()
    if return_rm:
        return res_df, p_w_rm
    else:
        return res_df


# TODO: add @time(info)
def retrieval_full(qids, index, queries, n_proc=Config.N_PROC, method='ql', results_df=None):
    if method == 'ql':
        result = _run_multiprocess_sync(run_ql_retrieval_process, qids, n_proc, index=index, queries=queries)
    elif method == 'rm':
        result = _run_multiprocess_sync(run_rm_retrieval_process, qids, n_proc, index=index, queries=queries)
    elif method == 'rm_rerank':
        result = _run_multiprocess_sync(run_rm_rerank_retrieval_process, qids, n_proc, index=index, queries=queries,
                                        results_df=results_df)
    else:
        logger.warn(f'Unknown method, choose between [ql, rm, rm_rerank]')
        return None
    df = pd.concat(result)
    logger.debug(df)
    return df


@timer  # TODO: add debug level
def initialize_text_index(text_inv, dict_txt, doc_lens, doc_names, index_globals):
    return IndexText(text_inverted=text_inv, terms_dict=dict_txt, index_global=index_globals, document_lengths=doc_lens,
                     document_names=doc_names)


@timer  # TODO: add debug level
def initialize_db_index(db_dir):
    return IndexDB(index_db_dir=db_dir)


@timer  # TODO: add debug level
def initialize_terrier_index(terrier_index_dir, **kwargs):
    return partial(IndexTerrier, terrier_index_dir=terrier_index_dir, **kwargs)


@timer  # TODO: add debug level
def initialize_ciff_queries(queries_file):
    return QueryParserCiff(queries_file)


@timer  # TODO: add debug level
def initialize_text_queries(queries_file):
    return QueryParserText(queries_file)
