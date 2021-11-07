import multiprocessing as mp
import os
from time import sleep

import lmdb
from tqdm import tqdm

from qpptk import Config, TermRecord, TermPosting, set_index_dump_paths, DocRecord, msgpack_encode
from qpptk.global_manager import initialize_text_index, initialize_ciff_index

logger = Config.get_logger()


def init_reader(*args):
    global index
    global queue
    index = args[0]
    queue = args[1]
    logger.debug(f'\n{mp.current_process()} with PID: {os.getpid()} has started\n')


def read_term_posting(term):
    global index
    global queue
    term_record: TermRecord = index.get_term_record(term)
    term_posting: TermPosting = index.get_posting_list(term)
    assert term_record.term == term_posting.term == term
    assert term_record.df == term_posting.df == len(term_posting.posting_list)
    assert term_record.cf == term_posting.cf
    queue.put(term_posting)


def read_doc_record(doc_id):
    global index
    global queue
    doc_record: DocRecord = index.doc_records.get(doc_id)
    # in CIFF index the doc_id is the locating in the ciff file
    doc_record = DocRecord(doc_id, doc_record.collection_doc_id, doc_record.doc_len)
    queue.put(doc_record)


def add_item(key, value, db, sdb):
    with db.begin(write=True, db=sdb) as txn:
        txn.put(key.encode(encoding='UTF-8'), msgpack_encode(value))


def writer(db_env, queue: mp.JoinableQueue, terms_len):
    logger.debug(f'{mp.current_process()} with PID: {os.getpid()} has started')
    with tqdm(total=terms_len) as progress:
        posting_db = db_env.open_db('postings_db'.encode())
        record_db = db_env.open_db('terms_db'.encode())
        docs_db = db_env.open_db('docs_db'.encode())
        while True:
            item = queue.get()
            if isinstance(item, TermPosting):
                add_item(item.term, item, db_env, posting_db)
                add_item(item.term, (item.cf, item.df), db_env, record_db)
            elif isinstance(item, DocRecord):
                add_item(str(item.doc_id), item, db_env, docs_db)
            queue.task_done()
            progress.update(1)


def _load_ciff_index():
    index_path = Config.CIFF_INDEX
    _prefix = index_path.rsplit('/', 1)[-1].replace('.ciff', '')
    _index = initialize_ciff_index(index_path)
    return _index, _prefix


def _load_text_index():
    index_path = Config.INDEX_DIR
    _prefix = '_'.join(index_path.split('/')[-2:])
    _index = initialize_text_index(*set_index_dump_paths(index_path))
    return _index, _prefix


def parse_index_to_db(index, prefix, db_dir):
    # TODO: Should add a map_size calculation
    db_env = lmdb.open(os.path.join(db_dir, prefix), create=True, subdir=True, map_size=2 ** 39, max_dbs=3)
    add_item('total_terms', index.total_terms, db_env, None)
    add_item('number_of_docs', index.number_of_docs, db_env, None)

    terms = index.terms_dict.keys()
    docs = index.doc_records.keys()
    processing_queue = mp.JoinableQueue()
    writer_proc = mp.Process(target=writer, args=(db_env, processing_queue, len(terms) + len(docs)), name='Writer',
                             daemon=True)
    writer_proc.start()
    with mp.Pool(processes=Config.N_PROC, initializer=init_reader, initargs=[index, processing_queue]) as pool:
        pool.map(read_term_posting, terms)
        logger.info('---- Finished Reading Terms ----')
        pool.map(read_doc_record, docs)
        logger.info('---- Finished Reading Docs----')
        processing_queue.join()
    logger.debug('The pool was terminated')

    writer_proc.terminate()
    sleep(2)
    logger.debug('Writer process was terminated')
    writer_proc.close()


if __name__ == '__main__':
    logger = Config.get_logger()
    num_processes = Config.N_PROC
    # num_processes = 30
    index, prefix = _load_ciff_index()
    # index, prefix = load_text_index()
    db_dir = Config.DB_DIR
    parse_index_to_db(index, prefix, db_dir)
