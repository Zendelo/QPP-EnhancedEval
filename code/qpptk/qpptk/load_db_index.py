import lmdb
import numpy as np
import os
import itertools as it

import pickle
from syct import timer
from scipy import sparse
from tqdm import tqdm

from qpptk import TermRecord, TermPosting, Config, msgpack_decode, DocRecord, ensure_file

logger = Config.get_logger()


class IndexDB:
    @classmethod
    def oov(cls, term):
        return TermPosting(term, 0, 0, tuple())  # Out of vocabulary terms

    def __init__(self, index_db_dir):
        self.db_env = lmdb.open(index_db_dir, create=False, subdir=True, map_size=2 ** 39, readonly=True, max_dbs=3,
                                lock=False)
        _stats = self.db_env.stat()
        logger.debug(f'Main DB stats: {_stats}')
        if _stats.get('overflow_pages'):
            logger.warning(f"The DB has {_stats.get('overflow_pages')} overflow_pages")
        self.posting_db = self.db_env.open_db('postings_db'.encode(), create=False)
        self.record_db = self.db_env.open_db('terms_db'.encode(), create=False)
        self.docs_db = self.db_env.open_db('docs_db'.encode(), create=False)
        with self.db_env.begin() as txn:
            self.total_terms = msgpack_decode(txn.get('total_terms'.encode()))
            self.number_of_docs = msgpack_decode(txn.get('number_of_docs'.encode()))
            _posting_db_stat = txn.stat(self.posting_db)
            _record_db_stat = txn.stat(self.record_db)
            _docs_db_stat = txn.stat(self.docs_db)
            logger.debug(f"Documents DB stats: {_docs_db_stat}")
        if _posting_db_stat.get('overflow_pages'):
            logger.warning(f"The postings DB has {_posting_db_stat.get('overflow_pages')} overflow_pages")
        assert _record_db_stat.get('entries') == _posting_db_stat.get('entries'), 'number of terms differs in named dbs'
        assert self.number_of_docs == _docs_db_stat.get('entries'), 'number of documents in db differs from total docs'
        self.unique_terms = _record_db_stat.get('entries')
        logger.debug(f"Vocab size {self.unique_terms}")
        self.index_db_dir = index_db_dir
        self.doc_terms_mat = None
        # one special key in the dict "reverse_mapping" holds a list of the terms by their indices
        self.terms_mapping_in_mat = None
        self.doc_len_vec = None
        self.doc_name_vec = None
        self.terms_cf_vec = None
        self._zero_len_docs = None
        self.doc_name_id_dict = None
        self.initialize_postings_sparse_mat()

    def get_terms_cf_vec(self, indices=None):
        if self.terms_cf_vec is None:
            self.terms_cf_vec = np.array(
                [v.cf for v in self.terms_mapping_in_mat.values() if isinstance(v, TermRecord)])
        if indices is not None:
            return self.terms_cf_vec[indices]
        else:
            return self.terms_cf_vec

    def get_posting_list(self, term: str) -> TermPosting:
        with self.db_env.begin(db=self.posting_db, write=False) as txn:
            term_posting = txn.get(term.encode())
            if term_posting:
                return TermPosting(*msgpack_decode(term_posting))
            else:
                return IndexDB.oov(term)

    # def __get_doc_record_db(self, doc_id):
    def get_doc_record(self, doc_id):
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
        with self.db_env.begin(db=self.record_db, write=False) as txn:
            term_record = txn.get(term.encode(), None)
            if term_record:
                term_record = TermRecord(term, np.nan, *msgpack_decode(term_record))
        return term_record

    def _generate_doc_len_vector(self):
        # docs_dict = {}
        doc_ids = []
        doc_names = []
        doc_lens = []
        zero_len_docs = []
        with self.db_env.begin(db=self.docs_db, write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                doc_id, doc_name, doc_len = msgpack_decode(value)
                doc_ids.append(doc_id)
                doc_names.append(doc_name)
                doc_lens.append(doc_len)
                if doc_len == 0:
                    zero_len_docs.append(doc_id)
                # docs_dict[doc_id] = doc_len
        doc_len_ar = np.array([*zip(doc_ids, doc_lens)], dtype=[('index', np.uint32), ('doc_len', np.uint32)])
        doc_name_ar = np.array([*zip(doc_ids, doc_names)], dtype=[('index', np.uint32), ('doc_name', object)])
        # doc_records = np.array(list(docs_dict.items()), dtype=[('index', np.uint32), ('doc_len', np.uint32)])
        doc_len_ar.sort(order='index')
        doc_name_ar.sort(order='index')
        self.doc_len_vec = doc_len_ar['doc_len']
        self.doc_name_vec = doc_name_ar['doc_name']
        if min(doc_ids) > 0:
            doc_ids = np.array(doc_ids) - 1
            zero_len_docs = np.array(zero_len_docs) - 1
        self._zero_len_docs = np.array(zero_len_docs)
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
    def initialize_postings_sparse_mat(self):
        doc_terms_mat_file = os.path.join(self.index_db_dir, 'doc_terms_mat.npz')
        terms_mapping_file = os.path.join(self.index_db_dir, 'terms_mapping.pkl')
        try:
            doc_terms_mat_file = ensure_file(doc_terms_mat_file)
            terms_mapping_file = ensure_file(terms_mapping_file)
            doc_terms_mat = sparse.load_npz(doc_terms_mat_file)
            self.doc_terms_mat = doc_terms_mat.astype(np.uint32)
            with open(terms_mapping_file, 'rb') as fp:
                self.terms_mapping_in_mat = pickle.load(fp)
        except FileNotFoundError:
            logger.info(f"doc_terms_mat.npz file wasn't found, will generate and save new ones")
            self.doc_terms_mat, self.terms_mapping_in_mat = self._generate_mat_from_terms()
            sparse.save_npz(doc_terms_mat_file, self.doc_terms_mat)
            with open(terms_mapping_file, 'wb') as fp:
                pickle.dump(self.terms_mapping_in_mat, fp)

    @timer
    def _generate_mat_from_terms(self):
        terms_mapping = {}
        reverse_terms_mapping = []
        row = []
        col = []
        data = []
        with tqdm(total=self.unique_terms) as progress:
            with self.db_env.begin(db=self.posting_db, write=False) as txn:
                i = 0
                for key, value in txn.cursor():
                    term, cf, df, posting_list = msgpack_decode(value)
                    terms_mapping[term] = TermRecord(term, i, cf, df)
                    reverse_terms_mapping.append(term)
                    doc_ids, tf_tuple = zip(*posting_list)
                    data.extend(tf_tuple)
                    col.extend([i] * df)
                    row.extend(doc_ids)
                    i += 1
                    progress.update(1)
        if min(row) > 0:
            row = [x - 1 for x in row]
        terms_mapping['reverse_mapping'] = reverse_terms_mapping
        return sparse.csr_matrix((data, (row, col)), shape=(self.number_of_docs, self.unique_terms),
                                 dtype=np.uint32), terms_mapping

    def get_full_mat(self):
        if self.doc_terms_mat is None:
            self.initialize_postings_sparse_mat()
        return self.doc_terms_mat

    def get_mat_by_terms(self, terms):
        self.get_full_mat()
        terms_indices = [self.terms_mapping_in_mat[t].id for t in terms if t in self.terms_mapping_in_mat]
        doc_terms_mat = self.doc_terms_mat[:, terms_indices]
        # return doc_terms_mat, dict(zip(terms, terms_indices))
        return doc_terms_mat

    def get_mat_by_docs(self, docs):
        self.get_full_mat()
        doc_terms_mat = self.doc_terms_mat[docs, :]
        return doc_terms_mat


if __name__ == '__main__':
    index_path = Config.CIFF_INDEX
    prefix = index_path.rsplit('/', 1)[-1].replace('.ciff', '')
    _db_dir = os.path.join(Config.DB_DIR, prefix)
    index = IndexDB(_db_dir)
    index.initialize_postings_sparse_mat()
