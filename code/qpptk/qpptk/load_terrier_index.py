import numpy as np
import os
import pyterrier as pt
from typing import Iterable
import multiprocessing as mp
from zlib import crc32

import pickle
from syct import timer
from scipy import sparse
from tqdm import tqdm

from qpptk import TermRecord, TermPosting, Config, msgpack_decode, DocRecord, ensure_file, ensure_dir, pickle_load_obj, \
    pickle_save_obj, Posting

logger = Config.get_logger()


class IndexTerrier:
    @classmethod
    def oov(cls, term):
        return TermPosting(term, 0, 0, tuple())  # Out of vocabulary terms

    def __str__(self):
        index_name = self.index_dir.rsplit('/', 1)[-1]
        if hasattr(self, 'partial_terms_hash'):
            index_name = f'{index_name}_partial-{self.partial_terms_hash}'
        return str(index_name)

    def __init__(self, terrier_index_dir, partial_terms=None, read_only=True, stats_index_path=None):
        # TODO: looks like I need to init terrier only if going to generate
        """
        The index is loading a Terrier index and several python objects that were generated from the Terrier index.
        To use it in a subprocess for retrieval first all the python objects should be generated and saved to disk,
        to generate the objects first time add read_only=False (only in the main process, should not run in parallel)
        """
        _generate = not read_only  # Added for brevity
        if mp.parent_process() and _generate:
            logger.warn("*** The generate attribute in the index is being used in a sub-process. ***")
            logger.warn("It should be used only in the main process, "
                        "during parallel runs the index should be used as read-only")
        self.index_dir = ensure_dir(terrier_index_dir)
        self.stats_index_dir = ensure_dir(stats_index_path if stats_index_path else self.index_dir)

        self.index_stats = self.__load_generate_stats_dict(_generate)
        self.total_terms = self.index_stats.get('total_terms')
        self.number_of_docs = self.index_stats.get('number_of_docs')
        self.unique_terms = self.index_stats.get('unique_terms')
        logger.debug(f"Vocab size {self.unique_terms}")
        self.terms_postings_mat = None  # row - doc_ids; col - term_ids
        self.docs_postings_mat = None
        self.terms_mapping_in_mat = None
        self.doc_len_vec = None
        self.doc_name_vec = None
        self.__terms_cf_vec = None
        self._zero_len_docs = None
        self.doc_name_id_dict = None
        self.terms_records = self.__load_generate_terms_dict(_generate)
        self.docs_records = self.__load_generate_docs_dict(_generate)
        if partial_terms is None:
            self.initialize_full_postings_sparse_mat(_generate)
        else:
            self.partial_terms_hash = crc32(''.join(partial_terms).encode())
            self.initialize_partial_postings_sparse_mat(partial_terms, _generate)

    @property
    def terms_cf_vec(self) -> np.ndarray:
        if self.__terms_cf_vec is None:
            self.__terms_cf_vec = np.array([(v.id, v.cf) for v in self.terms_records.values()],
                                           dtype=[('term_id', np.uint32), ('term_cf', np.uint32)])
            self.__terms_cf_vec.sort(order='term_id')
        return self.__terms_cf_vec

    def init_pt_index(self):
        # import and start pyterrier so that it works within tira and outside of tira
        from tira.third_party_integrations import ensure_pyterrier_is_loaded
        ensure_pyterrier_is_loaded()

        _index_file = ensure_file(os.path.join(self.index_dir, 'data.properties'))
        _indexref = pt.IndexRef.of(_index_file)
        return pt.IndexFactory.of(_indexref)

    def __generate_all_terms_records(self):
        terms_records = {}
        logger.info('Generating Terms Records Dict')
        with tqdm(total=self.unique_terms) as progress:
            for iterm in self._index.getLexicon():
                term = iterm.getKey()
                term_id = iterm.getValue().getTermId()
                df = iterm.getValue().getDocumentFrequency()
                cf = iterm.getValue().getFrequency()
                terms_records[term] = TermRecord(term, term_id, cf, df)
                progress.update(1)
        return terms_records

    def __load_generate_stats_dict(self, generate):
        stats_dict_file = os.path.join(self.stats_index_dir, 'stats_dict.pkl')
        if generate:
            _index = self.init_pt_index()
            _stats = _index.getCollectionStatistics().toString()
            logger.info(f'Terrier index stats:\n{_stats}')
            stats_dict = {'total_terms': _index.getCollectionStatistics().getNumberOfTokens(),
                          'number_of_docs': _index.getCollectionStatistics().getNumberOfDocuments(),
                          'unique_terms': _index.getCollectionStatistics().getNumberOfUniqueTerms()}
            pickle_save_obj(stats_dict, stats_dict_file)
            self._index = _index
        else:
            try:
                stats_dict_file = ensure_file(stats_dict_file)
                stats_dict = pickle_load_obj(stats_dict_file)
                logger.info('Terrier index stats:\n' + '\n'.join([f'{k}: {v}' for k, v in stats_dict.items()]))
            except FileNotFoundError as er:
                logger.warn(f'The file {stats_dict_file} can\'t be found, pass read_only=False to generate')
                raise er
        return stats_dict

    def __load_generate_terms_dict(self, generate):
        terms_dict_file = os.path.join(self.stats_index_dir, 'terms_dict.pkl')
        try:
            terms_dict_file = ensure_file(terms_dict_file)
            terms_records = pickle_load_obj(terms_dict_file)
        except FileNotFoundError as er:
            if generate:
                terms_records = self.__generate_all_terms_records()
                pickle_save_obj(terms_records, terms_dict_file)
            else:
                logger.warn(f'The file {terms_dict_file} can\'t be found')
                raise er
        return terms_records

    def __generate_all_docs_records(self):
        docs_records = {}
        logger.info('Generating Docs Records Dict')
        _doi = self._index.getDocumentIndex()
        with tqdm(total=self.number_of_docs) as progress:
            for doc_id in range(_doi.getNumberOfDocuments()):
                docs_records[doc_id] = DocRecord(doc_id, self._index.getMetaIndex().getItem("docno", doc_id),
                                                 _doi.getDocumentLength(doc_id))
                progress.update(1)
        return docs_records

    def __load_generate_docs_dict(self, generate):
        docs_records_file = os.path.join(self.stats_index_dir, 'docs_dict.pkl')
        try:
            docs_records_file = ensure_file(docs_records_file)
            docs_records = pickle_load_obj(docs_records_file)
        except FileNotFoundError as er:
            if generate:
                docs_records = self.__generate_all_docs_records()
                pickle_save_obj(docs_records, docs_records_file)
            else:
                logger.warn(f'The file {docs_records_file} can\'t be found')
                raise er
        return docs_records

    def get_terms_cf_vec(self, indices=None):
        return self.terms_cf_vec[indices]

    def get_posting_list(self, term: str) :
        # Warning to future self, Returning zip object instead of Post()
        term_id = self.terms_mapping_in_mat.get(term)
        if term_id is not None:
            posting = self.terms_postings_mat[:, term_id]
            return zip(posting.nonzero()[0], posting.data)
        else:
            return []

    # def __get_doc_record_db(self, doc_id):
    def get_doc_record(self, doc_id):  # FIXME
        with self.db_env.begin(db=self.docs_db, write=False) as txn:
            doc_record = txn.get(str(doc_id).encode())
        return DocRecord(*msgpack_decode(doc_record))

    # def get_doc_record(self, doc_id) -> DocRecord:
    #     return self.doc_records_cache.setdefault(doc_id, self.__get_doc_record_db(doc_id))

    def get_doc_len(self, doc_id: int) -> int:
        return self.get_doc_record(doc_id).doc_len

    def get_doc_name(self, doc_id: int) -> str:
        return self.get_doc_record(doc_id).collection_doc_id

    def get_term_cf(self, term: str) -> int:
        term_record = self.get_term_record(term)
        if term_record:
            return term_record.cf
        else:
            return 0

    def get_term_record(self, term: str) -> TermRecord:
        return self.terms_records.get(term)

    def _generate_doc_len_vector(self):  # FIXME
        # docs_dict = {}
        # doc_ids = []
        # doc_names = []
        # doc_lens = []
        # zero_len_docs = []
        doc_ids, doc_names, doc_lens = zip(*self.docs_records.values())
        doc_len_ar = np.array([*zip(doc_ids, doc_lens)], dtype=[('index', np.uint32), ('doc_len', np.uint32)])
        doc_name_ar = np.array([*zip(doc_ids, doc_names)], dtype=[('index', np.uint32), ('doc_name', object)])
        # doc_records = np.array(list(docs_dict.items()), dtype=[('index', np.uint32), ('doc_len', np.uint32)])
        doc_len_ar.sort(order='index')
        doc_name_ar.sort(order='index')
        self.doc_len_vec = doc_len_ar['doc_len']
        self.doc_name_vec = doc_name_ar['doc_name']
        # if min(doc_ids) > 0:
        #     doc_ids = np.array(doc_ids) - 1
        #     zero_len_docs = np.array(zero_len_docs) - 1
        # self._zero_len_docs = np.array(zero_len_docs)
        self.doc_name_id_dict = dict([*zip(doc_names, doc_ids)])
        return self.doc_len_vec, self.doc_name_vec, self.doc_name_id_dict

    def get_doc_len_vec(self, indices=None):
        doc_len = self.doc_len_vec if self.doc_len_vec is not None else self._generate_doc_len_vector()[0]
        if indices is not None:
            return doc_len[indices]
        else:
            return doc_len

    def get_doc_name_vec(self, indices=None):
        doc_name = self.doc_name_vec if self.doc_name_vec is not None else self._generate_doc_len_vector()[1]
        if indices is not None:
            return doc_name[indices]
        else:
            return doc_name

    def get_doc_ids_by_name(self, names_vec):
        self.get_doc_name_vec()
        return np.array([self.doc_name_id_dict.get(n) for n in names_vec])

    @timer
    def initialize_full_postings_sparse_mat(self, generate):
        terms_postings_mat_file = os.path.join(self.stats_index_dir, f'full_terms_postings_mat.npz')
        terms_mapping_file = os.path.join(self.stats_index_dir, f'full_terms_mapping.pkl')
        try:
            terms_mapping_file = ensure_file(terms_mapping_file)
            terms_postings_mat_file = ensure_file(terms_postings_mat_file)
            self.terms_postings_mat = pickle_load_obj(terms_postings_mat_file)
            self.terms_mapping_in_mat = pickle_load_obj(terms_mapping_file)
        except FileNotFoundError as ex:
            if generate:
                logger.info(f"{terms_mapping_file} or {terms_postings_mat_file} "
                            f"files are missing found, will generate and save new ones")
                self.terms_postings_mat, self.terms_mapping_in_mat = self._generate_mat_from_terms()
                sparse.save_npz(terms_postings_mat_file, self.terms_postings_mat)
                pickle_save_obj(self.terms_mapping_in_mat, terms_mapping_file)
            else:
                raise ex

    @timer
    def initialize_partial_postings_sparse_mat(self, partial_terms, generate):  # FIXME
        terms_postings_mat_file = os.path.join(self.stats_index_dir, f'partial_{self.partial_terms_hash}_postings_mat.npz')
        terms_mapping_file = os.path.join(self.stats_index_dir, f'partial_{self.partial_terms_hash}_mapping.pkl')
        try:
            terms_mapping_file = ensure_file(terms_mapping_file)
            terms_postings_mat_file = ensure_file(terms_postings_mat_file)
            self.terms_postings_mat = sparse.load_npz(terms_postings_mat_file)
            self.terms_mapping_in_mat = pickle_load_obj(terms_mapping_file)
        except FileNotFoundError as er:
            if generate:
                logger.info(f"{terms_mapping_file} or {terms_postings_mat_file} "
                            f"files are missing, will generate and save new ones")
                self.terms_postings_mat, self.terms_mapping_in_mat = self._generate_partial_mat_from_terms(
                    partial_terms)
                # logger.info(f"{terms_postings_mat_file} file is missing, will generate and save new one")
                # self.terms_postings_mat= self._generate_partial_mat_from_terms(partial_terms)
                sparse.save_npz(terms_postings_mat_file, self.terms_postings_mat)
                pickle_save_obj(self.terms_mapping_in_mat, terms_mapping_file)
            else:
                raise er

    @timer
    def _generate_mat_from_terms(self):  # FIXME
        terms_mapping = {}
        reverse_terms_mapping = []
        row = []
        col = []
        data = []
        lex = self._index.getLexicon()
        inv = self._index.getInvertedIndex()
        with tqdm(total=self.unique_terms) as progress:
            i = 0
            for iterm in lex:
                # posting_list = inv.getPostings(iterm.getValue())
                term = iterm.getKey()
                df = iterm.getValue().getDocumentFrequency()
                cf = iterm.getValue().getFrequency()
                terms_mapping[term] = TermRecord(term, i, cf, df)
                reverse_terms_mapping.append(term)
                doc_ids, tf_tuple = zip(*[(p.getId(), p.getFrequency()) for p in inv.getPostings(iterm.getValue())])
                data.extend(tf_tuple)
                col.extend([i] * df)
                row.extend(doc_ids)
                i += 1
                progress.update(1)
        terms_mapping['reverse_mapping'] = reverse_terms_mapping
        return sparse.csr_matrix((data, (row, col)), shape=(self.number_of_docs, self.unique_terms),
                                 dtype=np.uint32), terms_mapping

    @timer
    def _generate_partial_mat_from_terms(self, terms):
        terms_mapping = {}
        # reverse_terms_mapping = []
        row = []
        col = []
        data = []
        lex = self._index.getLexicon()
        inv = self._index.getInvertedIndex()

        for term in tqdm(terms):
            iterm = lex.getLexiconEntry(term)
            if iterm is None:
                logger.info(f'The term \"{term}\" is OOV')
                continue
            tid = iterm.getTermId()
            df = iterm.getDocumentFrequency()
            # cf = iterm.getFrequency()
            # terms_mapping[term] = TermRecord(term, tid, cf, df)
            terms_mapping[term] = tid
            doc_ids, tf_tuple = zip(*[(p.getId(), p.getFrequency()) for p in inv.getPostings(iterm)])
            data.extend(tf_tuple)
            col.extend([tid] * df)
            row.extend(doc_ids)
        return sparse.csr_matrix((data, (row, col)), shape=(self.number_of_docs, self.unique_terms),
                                 dtype=np.uint32), terms_mapping

    @timer
    def _generate_partial_mat_from_docs(self, docs):  # TODO: Implement here the docs index mat
        docs_mapping = {}
        # reverse_terms_mapping = []
        row = []
        col = []
        data = []
        lex = self._index.getLexicon()
        di = self._index.getDirectIndex()
        doi = self._index.getDocumentIndex()

        for doc in tqdm(docs):
            iterm = lex.getLexiconEntry(term)
            if iterm is None:
                logger.info(f'The term \"{term}\" is OOV')
                continue
            tid = iterm.getTermId()
            df = iterm.getDocumentFrequency()
            # cf = iterm.getFrequency()
            # docs_mapping[term] = TermRecord(term, tid, cf, df)
            docs_mapping[term] = tid
            doc_ids, tf_tuple = zip(*[(p.getId(), p.getFrequency()) for p in inv.getPostings(iterm)])
            data.extend(tf_tuple)
            col.extend([tid] * df)
            row.extend(doc_ids)
        return sparse.csr_matrix((data, (row, col)), shape=(self.number_of_docs, self.unique_terms),
                                 dtype=np.uint32)

    def get_mat_by_terms(self, terms: Iterable[str]):
        """
        :param terms - iterable of string terms
        :returns sparse index matrix
        """
        # terms_indices = [self.terms_mapping_in_mat[t] for t in terms if t in self.terms_mapping_in_mat]
        terms_indices = [
            self.terms_mapping_in_mat[t] if t in self.terms_mapping_in_mat else logger.warn(f'missing posting for {t}')
            for t in terms]
        return self.get_mat_by_term_ids(terms_indices)

    def get_mat_by_term_ids(self, terms_indices: Iterable[int]):
        """
        :param terms_indices - iterable of terms ids
        :returns sparse index matrix
        """
        doc_terms_mat = self.terms_postings_mat[:, terms_indices]
        return doc_terms_mat

    def get_mat_by_docs(self, docs):
        doc_terms_mat = self.terms_postings_mat[docs, :]
        return doc_terms_mat


if __name__ == '__main__':
    DEV_INDEX = '/research/local/olz/qpptk/dev_index/data.properties'
    # ROBUST_INDEX = '/research/local/olz/robust_krovetz_nostop_terrier/data.properties'
    # CW_INDEX = '/research/local/olz/cw12b_krovetz_nostop_terrier/index/krovetz-nostop/data.properties'
    # index_path = Config.CIFF_INDEX
    # prefix = index_path.rsplit('/', 1)[-1].replace('.ciff', '')
    # _db_dir = os.path.join(Config.DB_DIR, prefix)
    index = IndexTerrier(DEV_INDEX)
    index.initialize_full_postings_sparse_mat()
