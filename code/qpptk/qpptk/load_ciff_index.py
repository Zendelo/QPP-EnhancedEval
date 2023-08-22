from syct import timer

import qpptk.CommonIndexFileFormat_pb2 as ciff
from qpptk import TermPosting, Posting, TermRecord, DocRecord, read_message, Config

"""
An index stored in CIFF is a single file comprised of exactly the following:
    - A Header protobuf message,
    - Exactly the number of PostingsList messages specified in the num_postings_lists field of the Header
    - Exactly the number of DocRecord messages specified in the num_doc record field of the Header
 """

INDEX_CIFF_FILE = "/research/local/olz/ciff_indexes/robust04_Lucene_indri_krovetz.ciff"

# INDEX_CIFF_FILE = "/research/local/olz/ciff_indexes/cw12b/cw12b_Indri_nostop_krovetz.ciff"

logger = Config.logger


def parse_posting_list(posting_list: ciff.PostingsList) -> TermPosting:
    if posting_list:
        if not isinstance(posting_list, ciff.PostingsList):
            raise TypeError(f'parse_posting_lists is expecting a ciff.PostingsList {type(posting_list)} was passed')
    term, cf_t, df_t, posting_list = posting_list.term, posting_list.cf, posting_list.df, posting_list.postings
    docid, tf = posting_list[0].docid, posting_list[0].tf  # used later to translate gaps to docid
    _posting_list = [Posting(docid, tf)]
    for p in posting_list[1:]:
        docid += p.docid
        _posting_list.append(Posting(docid, p.tf))
    return TermPosting(term, cf_t, df_t, tuple(_posting_list))


@timer
def parse_index_file(index_file):
    terms_dict = {}
    doc_records = {}
    with open(index_file, 'rb') as fp:
        buf = fp.read()
        n = 0
        cur_n, header = read_message(buf, n, ciff.Header)
        print(f'header:\n{header}')
        num_postings_lists = header.num_postings_lists
        for _ in range(num_postings_lists):
            n, _posting_list = read_message(buf, cur_n, ciff.PostingsList)
            assert _posting_list.df == len(_posting_list.postings), f"The length of the posting list differs from df"
            terms_dict[_posting_list.term] = TermRecord(_posting_list.term, cur_n, _posting_list.cf, _posting_list.df)
            cur_n = n
        num_doc_records = header.num_docs
        for _ in range(num_doc_records):
            n, _doc_record = read_message(buf, cur_n, ciff.DocRecord)
            doc_records[_doc_record.docid] = DocRecord(cur_n, _doc_record.collection_docid, _doc_record.doclength)
            cur_n = n
    return header, terms_dict, doc_records


# TODO: add validation using the header message, without saving the header - should be pickable
class IndexCiff:
    @classmethod
    def oov(cls, term):
        return TermPosting(term, 0, 0, tuple())  # Out of vocabulary terms

    def __init__(self, header, index_file, terms_dict, doc_records):
        sum_terms = sum(map(lambda x: x.doc_len, doc_records.values()))
        if sum_terms != header.total_terms_in_collection:
            logger.warn(f'total_terms_in_collection stat in the header != sum of all doc_lens')
            logger.warn(f'header.total_terms_in_collection: {header.total_terms_in_collection}')
            logger.warn(f'sum of all documents lengths: {sum_terms}')
        self.total_terms = sum_terms
        assert header.total_docs == len(doc_records), f'total docs in header differs from total docs in doc_records'
        self.number_of_docs = header.total_docs
        with open(index_file, 'rb') as fp:
            self.file_buf = fp.read()
        self.terms_dict = terms_dict
        self.doc_records = doc_records

    def _get_raw_posting_list(self, term_id):
        return self._read_index_line(term_id.id)

    def _read_index_line(self, n):
        _, posting_list = read_message(self.file_buf, n, ciff.PostingsList)
        return posting_list

    def get_posting_list(self, term: str) -> TermPosting:
        term_id = self.terms_dict.get(term)
        if term_id:
            raw_posting_lists = self._get_raw_posting_list(term_id)
            return parse_posting_list(raw_posting_lists)
        else:
            return IndexCiff.oov(term)

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
    index = IndexCiff(header, INDEX_CIFF_FILE, terms_dict, doc_records)
    x = index.get_posting_list('ingathering')
    print(len(x.posting_list))


if __name__ == '__main__':
    header, terms_dict, doc_records = parse_index_file(INDEX_CIFF_FILE)
    main()
