from collections import defaultdict

import pandas as pd
from syct import timer
from lxml import etree

import qpptk.CommonIndexFileFormat_pb2 as ciff
from qpptk import Config, read_message, DocRecord, transform_list_to_counts_dict, add_topic_to_qdf


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
    def __init__(self, queries_file, **kwargs):
        header, queries_dict, query_records = parse_ciff_queries_file(queries_file)
        self.queries_dict = queries_dict
        self._rename_queries_to_qid(query_records)
        filter_queries_file: list = kwargs.get('filter_queries_file', [])
        if filter_queries_file:
            self.filter_queries = QueryParserText(filter_queries_file).get_query_ids()
            self._filter_queries()

    def _rename_queries_to_qid(self, query_records):
        """
        Translates the query keys (qid) from index id (given during indexing) to collection id (original qid)
        :return:
        """
        for iid in self.get_query_ids():
            self.queries_dict[query_records[iid].collection_doc_id] = self.queries_dict.pop(iid)

    def _filter_queries(self):
        filter_qids = set(self.filter_queries).intersection(self.queries_dict.keys())
        self.queries_dict = {qid: self.queries_dict.get(qid) for qid in filter_qids}

    def get_query(self, qid: str) -> dict:
        """
        get the query dict for a given qid if it exists, otherwise returns None
        :param qid:
        :return: dict {'international': 1, 'organized': 1, 'crime': 1}
        """
        return self.queries_dict.get(qid)

    def get_query_ids(self) -> list:
        """
        get the query a sorted list of query ids
        :return: list of sorted query ids ['301', '302', '303']
        """
        return sorted(self.queries_dict.keys())


class QueryParserText:
    def __init__(self, queries_txt_file, **kwargs):
        self.queries_file = queries_txt_file
        self.raw_queries_df = self._read_queries()
        self.queries_sr = self._weight_queries()
        filter_queries_file: list = kwargs.get('filter_queries_file', [])
        if filter_queries_file:
            self.filter_queries = self._read_queries(filter_queries_file).index
            self._filter_queries()

    def _read_queries(self, queries_file=None):
        _queries_file = queries_file if queries_file else self.queries_file
        with open(_queries_file, 'r') as fp:
            queries = [line.strip().split(' ', maxsplit=1) for line in fp]
        return pd.DataFrame(queries, columns=['qid', 'terms']).set_index('qid')

    def _weight_queries(self):
        return self.raw_queries_df.terms.str.split().apply(transform_list_to_counts_dict)

    def _filter_queries(self):
        self.queries_sr = self.queries_sr.loc[self.filter_queries]

    def get_duplicates_bow(self):
        duplicate_qids = set()
        for topic, _df in add_topic_to_qdf(pd.DataFrame(self.queries_sr)).set_index('qid').groupby('topic')['terms']:
            try:
                duplicate_qids.update(_df.loc[_df.duplicated('last')].index)
                # duplicate_qids.update(set(_df.index).difference(_df.drop_duplicates('last').index))
            except SystemError as err:
                _df = add_topic_to_qdf(self.raw_queries_df).set_index('topic').loc[topic].set_index('qid', drop=True)
                duplicate_qids.update(_df.loc[_df['terms'].str.split().map(sorted).map(tuple).duplicated('last')].index)
        return duplicate_qids

    def get_duplicates_seq(self):
        duplicate_qids = set()
        for topic, _df in add_topic_to_qdf(self.raw_queries_df).set_index('qid').groupby('topic')['terms']:
            duplicate_qids.update(duplicate_qids.update(_df.loc[_df.duplicated('last')].index))
        return duplicate_qids

    def get_query(self, qid: str) -> dict:
        return self.queries_sr.loc[qid]

    def get_query_ids(self) -> list:
        return self.queries_sr.index.tolist()


class QueriesXMLWriter:
    def __init__(self, queries_df: pd.DataFrame):
        self.queries_df = queries_df.apply(' '.join)
        self.root = etree.Element('parameters')
        self._add_queries()

    def _add_queries(self):
        for qid, text in self.queries_df.iteritems():
            query = etree.SubElement(self.root, 'query')
            number = etree.SubElement(query, 'number')
            number.text = qid
            txt = etree.SubElement(query, 'text')
            txt.text = '#combine( {} )'.format(text)

    def print_queries_xml(self):
        """Prints to STD.OUT (usually the screen)"""
        print(etree.tostring(self.root, pretty_print=True, encoding='unicode'))

    def print_queries_xml_file(self, file_name):
        """Prints to a File"""
        print(etree.tostring(self.root, pretty_print=True, encoding='unicode'), file=open(file_name, 'w'))


@timer
def test():
    txt_queries = '/research/local/olz/data/robust04.qry'
    txt_obj = QueryParserText(txt_queries)
    QueriesXMLWriter(txt_obj.raw_queries_sr).print_queries_xml_file('/research/local/olz/data/robust04.qry.xml')


if __name__ == '__main__':
    filter_queries = QueryParserText('/research/local/olz/data/core17.qry')
    with open('duplicated_qids.txt', 'w') as f:
        for qid in sorted(QueryParserText('../robust-uqv-orig.qry').get_duplicates_bow()):
            f.write(qid + '\n')

    qp = QueryParserCiff(Config.CIFF_QUERIES, filter_queries=filter_queries.get_query_ids())
    # print(qp.get_query_ids())
    # print(qp.get_query(qp.get_query_ids()[0]))

    # test()
