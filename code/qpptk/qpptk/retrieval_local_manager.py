import numpy as np
from scipy.sparse import csr_matrix

from qpptk import IndexDB as Index
from qpptk import QueryParserText, Config

logger = Config.logger


class LocalManagerRetrieval:
    def __init__(self, index_obj: Index, query_obj: QueryParserText, qid):
        self.qid = qid
        self.mu = Config.MU
        self.num_docs = Config.NUM_DOCS
        self.fb_docs = Config.FB_DOCS
        self.fb_terms = Config.FB_TERMS
        self.query = query_obj.get_query(qid)
        self.index = index_obj
        self.qry_terms_record = {q: index_obj.get_term_record(q) for q in self.query if index_obj.get_term_record(q)}
        self.oov_terms = set()
        self._candidates_dict = {}

    def _ql_score_documents(self):
        tf_mat = self.index.get_mat_by_terms(self.qry_terms_record.keys())
        nnz_rows = np.unique(tf_mat.nonzero()[0])
        tf_mat = tf_mat[nnz_rows, :]
        _doc_len = self.index.get_doc_len_vec(nnz_rows)
        _left_log = tf_mat.multiply(csr_matrix(1 / (self.mu + _doc_len)).T)
        _right_log = np.array([1 - _doc_len / (self.mu + _doc_len), ]).T * np.array(
            [[v.cf for v in self.qry_terms_record.values()], ]) / self.index.total_terms
        doc_scores = np.asarray(
            np.log(_left_log + _right_log) * np.array([[self.query[q] for q in self.qry_terms_record], ]).T).squeeze()
        return np.sort(
            np.array(list(zip(nnz_rows, doc_scores)), dtype=[('doc_id', np.uint32), ('doc_score', np.float)]),
            order='doc_score')[::-1]

    def rm_construction(self, ql_top_k):
        """
        Constructing the RM without smoothing
        """
        tf_mat = self.index.get_mat_by_docs(ql_top_k['doc_id'])
        nnz_cols = np.unique(tf_mat.nonzero()[1])
        tf_mat = tf_mat[:, nnz_cols]
        exp_ql = np.exp(ql_top_k['doc_score'])
        sum_exp_ql = exp_ql.sum()
        _doc_len = self.index.get_doc_len_vec(ql_top_k['doc_id'])
        _p_w_rm = np.array([exp_ql / _doc_len, ]) * tf_mat
        p_w_rm = np.asarray(_p_w_rm / sum_exp_ql).squeeze()
        return np.sort(np.array(list(zip(nnz_cols, p_w_rm)), dtype=[('term_id', np.uint32), ('term_score', np.float)]),
                       order='term_score')[::-1]

    def rank_by_kl(self, p_w_rm_top_n, rank_doc_ids=None):
        """
        Ranking documents with -cross entropy (equivalent to KL divergence), using p_w_rm_top_n terms from RM
        """
        if rank_doc_ids is not None:
            tf_mat = self.index.get_mat_by_docs(rank_doc_ids['doc_id'])[:, p_w_rm_top_n['term_id']]
            _doc_len = self.index.get_doc_len_vec(rank_doc_ids['doc_id'])
            doc_indices = rank_doc_ids['doc_id']
        else:
            tf_mat = self.index.get_full_mat()[:, p_w_rm_top_n['term_id']]
            _doc_len = self.index.get_doc_len_vec()
            nnz_rows = np.unique(tf_mat.nonzero()[0])
            tf_mat = tf_mat[nnz_rows, :]
            _doc_len = _doc_len[nnz_rows]
            doc_indices = nnz_rows
        terms_cf = self.index.get_terms_cf_vec(p_w_rm_top_n['term_id'])
        _left = tf_mat.multiply(csr_matrix(1 / (self.mu + _doc_len)).T)
        _right = np.array([1 - _doc_len / (self.mu + _doc_len), ]).T * np.array([terms_cf, ]) / self.index.total_terms
        doc_lm = np.log(_left + _right)
        scored_docs = np.asarray(doc_lm.dot(p_w_rm_top_n['term_score'].T)).squeeze()
        return np.sort(
            np.array(list(zip(doc_indices, scored_docs)), dtype=[('doc_id', np.uint32), ('doc_score', np.float)]),
            order='doc_score')[::-1]

    def _check_if_query_oov(self):
        if len(self.query) == len(self.oov_terms):
            logger.warning(f'---- The entire query {self.qid} is OOV!!! ----')
            return True
        return False

    def _generate_matching_postings(self):
        for term in self.query:
            term, cf, df, posting_list = self.index.get_posting_list(term)
            if cf == 0:
                self.oov_terms.add(term)
                continue
            self._candidates_dict[term] = dict(posting_list)

    def translate_doc_id_to_doc_no(self, result_tuple):
        return tuple(map(lambda x: (self.index.get_doc_name(x[0]), x[1]), result_tuple))

    def translate_doc_id_to_doc_no_vec(self, result_tuple):
        doc_name_vec = self.index.get_doc_name_vec()
        return tuple(zip(doc_name_vec[result_tuple['doc_id']], result_tuple['doc_score']))

    def translate_doc_no_to_doc_id_vec(self, result_tuple):
        return tuple(zip(self.index.get_doc_ids_by_name(result_tuple['doc_no']), result_tuple['doc_score']))

    def _is_empty_query(self):
        if self._check_if_query_oov():
            return True
        if self.oov_terms:
            logger.info(f"Query: {self.qid}; terms {self.oov_terms} are out of vocabulary")
        return False

    def _init_sorted_ql_docs(self, working_set_docs=None):
        if self._is_empty_query():
            return tuple()
        if working_set_docs.any():
            doc_scores = working_set_docs.T[1]
            doc_ids = self.index.get_doc_ids_by_name(working_set_docs.T[0])
            sorted_ql_docs = np.array(list(zip(doc_ids, doc_scores)),
                                      dtype=[('doc_id', np.uint32), ('doc_score', np.float)])
        else:
            sorted_ql_docs = self._ql_score_documents()
        return sorted_ql_docs

    def run_ql_retrieval(self, working_set_docs=None):
        # TODO: add option for working set docs
        if self._is_empty_query():
            return tuple()
        # self._generate_matching_postings()
        sorted_ql_scored_docs = self._ql_score_documents()
        return self.translate_doc_id_to_doc_no_vec(sorted_ql_scored_docs[:self.num_docs])

    def run_rm_retrieval(self, re_rank_ql=False, working_set_docs=None):
        """
        This method will use QL to retrieve an initial set of documents, then use the set to create a RM (RM1)
        if the re_rank_ql param is True, it will use the RM to re-rank the top QL documents with the RM.
        Otherwise, it will rank all the documents in the corpus.
        """
        sorted_ql_docs = self._init_sorted_ql_docs(working_set_docs)
        sorted_rm_terms = self.rm_construction(sorted_ql_docs[:self.fb_docs])
        if re_rank_ql or working_set_docs:
            sorted_rm_scored_docs = self.rank_by_kl(sorted_rm_terms[:self.fb_terms], sorted_ql_docs[:self.num_docs])
        else:
            sorted_rm_scored_docs = self.rank_by_kl(sorted_rm_terms[:self.fb_terms])
        return self.translate_doc_id_to_doc_no_vec(sorted_rm_scored_docs[:self.num_docs]), sorted_rm_terms

    def generate_rm(self, working_set_docs):
        """
        This method will use QL to
        """
        sorted_ql_docs = self._init_sorted_ql_docs(working_set_docs)
        return self.rm_construction(sorted_ql_docs)
