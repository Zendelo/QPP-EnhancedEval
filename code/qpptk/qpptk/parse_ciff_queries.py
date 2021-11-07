from collections import defaultdict

from syct import timer

import qpptk.CommonIndexFileFormat_pb2 as ciff
from qpptk import Config, read_message, DocRecord


@timer
def parse_ciff_queries_file(queries_file):
    queries_dict = defaultdict(dict)
    query_records = {}
    with open(queries_file, 'rb') as fp:
        buf = fp.read()
        n = 0
        cur_n, header = read_message(buf, n, ciff.Header)
        num_postings_lists = header.num_postings_lists
        for _ in range(num_postings_lists):
            # TODO: add sanity check that df equals to len of posting
            n, _posting_list = read_message(buf, cur_n, ciff.PostingsList)
            term, df, cf, posting = _posting_list.term, _posting_list.df, _posting_list.cf, _posting_list.postings
            prev_docid = 0
            for p in posting:
                docid = p.docid + prev_docid
                queries_dict[docid].update({term: p.tf})
                prev_docid = docid
            cur_n = n
        num_doc_records = header.num_docs
        for _ in range(num_doc_records):
            n, _doc_record = read_message(buf, cur_n, ciff.DocRecord)
            query_records[_doc_record.docid] = DocRecord(cur_n, _doc_record.collection_docid, _doc_record.doclength)
            cur_n = n
    return header, queries_dict, query_records


class QueryParserCiff:
    def __init__(self, queries_dict, query_records, **kwargs):
        self.queries_dict = queries_dict
        self.query_records = query_records
        self._rename_queries_to_qid()
        self.filter_topics: list = kwargs.get('filter_queries', [])
        if self.filter_topics:
            self._filter_topics()

    def _rename_queries_to_qid(self):
        """
        Translates the query keys (qid) from index id (given during indexing) to collection id (original qid)
        :return:
        """
        for iid in self.get_query_ids():
            self.queries_dict[self.query_records[iid].collection_doc_id] = self.queries_dict.pop(iid)

    def _filter_topics(self):
        pass

    def get_query(self, qid: str) -> dict:
        # {'international': 1, 'organized': 1, 'crime': 1}
        return self.queries_dict.get(qid)

    def get_query_ids(self) -> list:
        # ['301', '302', '303']
        return sorted(self.queries_dict.keys())


if __name__ == '__main__':
    header, queries_dict, query_records = parse_ciff_queries_file(Config.CIFF_QUERIES)
    qp = QueryParserCiff(queries_dict, query_records)
    print(qp.get_query_ids())
    print(qp.get_query(qp.get_query_ids()[0]))
