import numpy as np

# from qpptk import IndexDB as Index
from qpptk import IndexTerrier as Index


class LocalManagerPredictorPre:
    def __init__(self, index_obj: Index, query_obj, qid):
        self.index = index_obj
        self.queries = query_obj
        self.query = query_obj.get_query(qid)
        self.total_docs = index_obj.number_of_docs
        self.terms_df = self._generate_terms_df_vec()
        self.terms_cf = self._generate_terms_cf_vec()
        self.raw_scq = np.array([])
        self.raw_var = np.array([])

    def _generate_terms_df_vec(self):
        """returns a tuple of df for each term in the query (df_t1, df_t2, ... , df_tn)"""
        return tuple(self.index.get_term_record(tr).df for tr in self.query if self.index.get_term_record(tr))

    def _generate_terms_cf_vec(self):
        """returns a tuple of df for each term in the query (df_t1, df_t2, ... , df_tn)"""
        return tuple(self.index.get_term_record(tr).cf for tr in self.query if self.index.get_term_record(tr))

    def calc_max_idf(self):
        """
        Scholer, F. et al. 2004. Query association surrogates for Web search.
        """
        return np.log(np.array(self.total_docs) / self.terms_df).max()

    def calc_avg_idf(self):
        """
        Cronen-Townsend, S. et al. 2002. Predicting query performance.
        """
        return np.log(np.array(self.total_docs) / self.terms_df).mean()

    def _calc_raw_scq(self):
        """
        Zhao, Y. et al. 2008.
        Effective Pre-retrieval Query Performance Prediction Using Similarity and Variability Evidence
        """
        self.raw_scq = (1 + np.log(self.terms_cf)) * np.log(1 + np.array(self.total_docs) / self.terms_df)
        return self.raw_scq

    def calc_scq(self):
        """
        Zhao, Y. et al. 2008.
        Effective Pre-retrieval Query Performance Prediction Using Similarity and Variability Evidence
        """
        return self.raw_scq.sum() if self.raw_scq.any() else self._calc_raw_scq().sum()

    def calc_max_scq(self):
        """
        Zhao, Y. et al. 2008.
        """
        return self.raw_scq.max() if self.raw_scq.any() else self._calc_raw_scq().max()

    def calc_avg_scq(self):
        """
        Zhao, Y. et al. 2008.
        In the original paper it's described as NSCQ
        """
        return self.raw_scq.mean() if self.raw_scq.any() else self._calc_raw_scq().mean()

    def _calc_w_d_t(self, term):
        term_record = self.index.get_term_record(term)
        if not term_record:
            # If the term is OOV, the w_d_t is set to 0
            return 0
        term_posting = dict(self.index.get_posting_list(term))
        tf_vec = np.fromiter(term_posting.values(), dtype=int)
        assert len(term_posting) == term_record.df, f'For term: "{term}" the number of docs in posting different from df'
        return 1 + np.log(tf_vec) * np.log(1 + self.total_docs / term_record.df)

    def _calc_raw_var(self):
        """
        Zhao, Y. et al. 2008.
        Effective Pre-retrieval Query Performance Prediction Using Similarity and Variability Evidence
        """
        result = []
        for term in self.query:
            result.append(np.std(self._calc_w_d_t(term)))
        self.raw_var = np.array(result)
        return self.raw_var

    def calc_var(self):
        """
        Zhao, Y. et al. 2008.
        """
        return self.raw_var.sum() if self.raw_var.any() else self._calc_raw_var().sum()

    def calc_max_var(self):
        """
        Zhao, Y. et al. 2008.
        """
        return self.raw_var.max() if self.raw_var.any() else self._calc_raw_var().max()

    def calc_avg_var(self):
        """
        Zhao, Y. et al. 2008.
        """
        return self.raw_var.mean() if self.raw_var.any() else self._calc_raw_var().mean()
