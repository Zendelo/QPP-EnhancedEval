import os

import pyterrier as pt
from syct import timer
from tqdm import tqdm

from qpptk import TermPosting, Posting, TermRecord, DocRecord, Config, ensure_dir, ensure_file

"""
A Terrier index python interface
 """

# DEV_INDEX = '/research/local/olz/qpptk/dev_index/data.properties'
# ROBUST_INDEX = '/research/local/olz/robust_krovetz_nostop_terrier/data.properties'
CW_INDEX = '/research/local/olz/cw12b_krovetz_nostop_terrier/index/krovetz-nostop/data.properties'

logger = Config.logger


class IndexTerri:

    @classmethod
    def oov(cls, term):
        return TermPosting(term, 0, 0, tuple())  # Out of vocabulary terms

    def __init__(self, terrier_index_dir):
        self.index_dir = ensure_dir(terrier_index_dir)
        _index_file = ensure_file(os.path.join(self.index_dir, 'data.properties'))
        if not pt.started():
            pt.init()
            pt.logging('INFO')
        _indexref = pt.IndexRef.of(_index_file)
        self.index = pt.IndexFactory.of(_indexref)
        _stats = self.index.getCollectionStatistics().toString()
        logger.debug(f'Terrier index stats:\n{_stats}')
        self.total_terms = self.index.getCollectionStatistics().getNumberOfTokens()
        self.number_of_docs = self.index.getCollectionStatistics().getNumberOfDocuments()
        self.unique_terms = self.index.getCollectionStatistics().getNumberOfUniqueTerms()
        self._lex = self.index.getLexicon()
        self._inv = self.index.getInvertedIndex()
        self._doi = self.index.getDocumentIndex()
        self._meta = self.index.getMetaIndex()
        self.terms_mapping = None
        self.docs_dict = None

    @property
    def terms_dict(self):
        if self.terms_mapping:
            return self.terms_mapping
        terms_mapping = {}
        logger.debug('Generating Terms Dict')
        with tqdm(total=self.unique_terms) as progress:
            for iterm in self._lex:
                term = iterm.getKey()
                term_id = iterm.getValue().getTermId()
                df = iterm.getValue().getDocumentFrequency()
                cf = iterm.getValue().getFrequency()
                terms_mapping[term] = TermRecord(term, term_id, cf, df)
                progress.update(1)
        self.terms_mapping = terms_mapping
        return self.terms_mapping

    @property
    def doc_records(self):
        if self.docs_dict:
            return self.docs_dict
        docs_mapping = {}
        logger.debug('Generating Docs Dict')
        with tqdm(total=self.number_of_docs) as progress:
            for doc_id in range(self._doi.getNumberOfDocuments()):
                docs_mapping[doc_id] = DocRecord(doc_id, self._meta.getItem("docno", doc_id),
                                                 self._doi.getDocumentLength(doc_id))
                progress.update(1)
        self.docs_dict = docs_mapping
        return self.docs_dict

    def get_posting_list(self, term: str) -> TermPosting:
        if not pt.started():
            pt.init()
        term_record = self.terms_dict.get(term)
        if term_record:
            _posting_list = [Posting(p.getId(), p.getFrequency()) for p in
                             self._inv.getPostings(self._lex.getLexiconEntry(term))]
            return TermPosting(term, term_record.cf, term_record.df, tuple(_posting_list))
        else:
            return IndexTerri.oov(term)

    def get_doc_len(self, doc_id: int) -> int:
        return self.doc_records.get(doc_id).doc_len

    def get_doc_name(self, doc_id: int) -> str:
        return self.doc_records.get(doc_id).collection_doc_id

    def get_term_cf(self, term: str) -> float:
        return self.terms_dict.get(term).cf

    def get_term_record(self, term: str) -> TermRecord:
        return self.terms_dict.get(term)


@timer
def main():
    index = IndexTerri(CW_INDEX)
    x = index.get_posting_list('gather')
    print(len(x.posting_list))
    print(index.get_doc_len(0))
    print(index.terms_dict.get('gather'))
    print(index.terms_dict.get('crime'))
    print(index.doc_records.get(100))
    print(index.doc_records.get(101))


if __name__ == '__main__':
    main()
