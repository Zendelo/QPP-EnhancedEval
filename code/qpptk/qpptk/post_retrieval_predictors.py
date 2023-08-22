import numpy as np

from qpptk import IndexDB as Index


class LocalManagerPredictorPost:
    def __init__(self, index_obj: Index, query_obj, qid, results_df):
        self.index = index_obj
        self.results_df = results_df
        self.queries = query_obj
        self.query = query_obj.get_query(qid)
        self.scores_vec = self._init_scores_vec(qid)
        self._drop_oov_query_terms()
        self.qid = qid
        self.ql_corpus_score = self._calc_corpus_score()

    def _drop_oov_query_terms(self):
        self.query = {k: v for k, v in self.query.items() if self.index.get_term_record(k)}

    def _init_scores_vec(self, qid, results_df=None):
        if results_df is None:
            results_df = self.results_df
        return np.array(results_df.loc[qid]['docScore'].tolist())

    def _calc_corpus_score(self):
        cf_vec = np.array([self.index.get_term_cf(term) for term in self.query])
        return np.log(cf_vec / self.index.total_terms).dot(np.array(list(self.query.values()), dtype=np.uint8))

    def calc_wig(self, list_size_param):
        """
        Y. Zhou and W. B. Croft. Query performance prediction in web search environments
        """
        try:
            scores_vec = self.scores_vec[:list_size_param]
        except IndexError as err:
            print(err)
            print(self.qid)
        return (scores_vec.mean() - self.ql_corpus_score) / np.sqrt(len(self.query))

    def calc_nqc(self, list_size_param):
        """
        A. Shtok, O. Kurland, and D. Carmel. Predicting query performance by query-drift estimation
        """
        scores_vec = self.scores_vec[:list_size_param]
        return np.std(scores_vec) / abs(self.ql_corpus_score)

    def calc_smv(self, list_size_param):
        """
        Y. Tao and S. Wu. Query performance prediction by considering score magnitude and variance together
        """
        scores_vec = self.scores_vec[:list_size_param]
        mean = scores_vec.mean()
        return (scores_vec * abs(np.log(scores_vec / mean))).mean() / self.ql_corpus_score

    def calc_clarity(self, p_w_rm):
        """
        S. Cronen-Townsend, Y. Zhou, and W. Bruce Croft. Predicting query performance
        """
        terms_cf = self.index.get_terms_cf_vec(p_w_rm['term_id'])
        _p_w_rm = p_w_rm['term_score'] / p_w_rm['term_score'].sum()
        corpus_lm = terms_cf['term_cf'] / self.index.total_terms
        return np.log(_p_w_rm / corpus_lm).dot(_p_w_rm)

    def calc_dfr_info_bo2(self, p_w_rm):
        # FIXME: Need to construct the RM* based on the top k=(10) documents
        """
        Amati Giambattista, Carpineto Claudio and Romano Giovanni.
        Query Difficulty, Robustness, and Selective Application of Query Expansion
        """
        terms_cf = self.index.get_terms_cf_vec(p_w_rm['term_id'])
        _p_w_rm = p_w_rm['term_score'] / p_w_rm['term_score'].sum()
        corpus_lm = np.array([terms_cf, ]) / self.index.total_terms
        return np.log(_p_w_rm / corpus_lm).dot(_p_w_rm)[0]

    def calc_uef(self, list_size_param, rm_results_df, predictor_result):
        """
        A. Shtok, O. Kurland, and D. Carmel.
        Using statistical decision theory and relevance models for query-performance prediction.
        """
        results_df = self.results_df.loc[self.qid].head(list_size_param)
        similarity = results_df.loc[self.qid, ['docNo', 'docScore']].set_index('docNo', drop=True).corrwith(
            rm_results_df.loc[:, ['docNo', 'docScore']].set_index('docNo', drop=True))[0]
        return similarity * predictor_result

    def calc_qf(self, list_size_param, rm_results_df):
        """
        Zhou, Yun and Croft, W Bruce. Query Performance Prediction in Web Search Environments.
        """
        original_docs_set = set(self.results_df.loc[self.qid].head(list_size_param).docNo)
        overlap = len(original_docs_set.intersection(rm_results_df.head(list_size_param).docNo)) / list_size_param
        return overlap
