import logging
import os
from typing import NamedTuple, Tuple

import toml

from qpptk import ensure_dir, ensure_file

# def __init_logger(self, logger):
#     if logger:
#         return logger
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     if not logger.hasHandlers():
#         formatter = logging.Formatter('{asctime} - {message}', datefmt="%H:%M:%S", style="{")
#         handler = logging.StreamHandler()
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)
#     return logger

CONFIG_FILE = os.path.dirname(os.path.realpath(__file__)) + '/config.toml'


def set_index_dump_paths(index_dir):
    text_inv = ensure_file(os.path.join(index_dir, 'text.inv'))
    dict_txt = ensure_file(os.path.join(index_dir, 'dict_new.txt'))
    doc_lens = ensure_file(os.path.join(index_dir, 'doc_lens.txt'))
    doc_names = ensure_file(os.path.join(index_dir, 'doc_names.txt'))
    index_globals = ensure_file(os.path.join(index_dir, 'global.txt'))
    return text_inv, dict_txt, doc_lens, doc_names, index_globals


class Config:
    config_file = ensure_file(CONFIG_FILE)
    config = toml.load(config_file)

    # Defaults

    parameters = config.get('parameters')
    MU = parameters.get('mu')
    FB_DOCS = parameters.get('fb_docs')
    # The maximum number of documents to use for the re-ranking, comment out to re-rank all docs in initial list
    WORKING_SET_SIZE = parameters.get('working_set_size', None)
    FB_TERMS = parameters.get('fb_terms')
    NUM_DOCS = parameters.get('max_result_size')

    N_PROC = parameters.get('num_processes', 1)

    prediction_parameters = parameters.get('prediction')
    WIG_LIST_SIZE = prediction_parameters.get('wig_list_size')
    NQC_LIST_SIZE = prediction_parameters.get('nqc_list_size')
    SMV_LIST_SIZE = prediction_parameters.get('smv_list_size')

    CLARITY_FB_TERMS = prediction_parameters.get('clarity_fb_terms')
    CLARITY_LIST_SIZE = prediction_parameters.get('clarity_list_size')

    QF_FB_TERMS = prediction_parameters.get('qf_fb_terms')
    QF_LIST_SIZE = prediction_parameters.get('qf_list_size')
    QF_OVERLAP_SIZE = prediction_parameters.get('qf_overlap_size')

    UEF_FB_TERMS = prediction_parameters.get('uef_fb_terms')
    UEF_LIST_SIZE = prediction_parameters.get('uef_list_size')
    UEF_RANKING_SIZE = prediction_parameters.get('uef_ranking_size')

    logging_level = parameters.get('logging_level', 'DEBUG')
    # logging_level = logging.DEBUG

    # uef_parameters = prediction_parameters.get('uef')
    # UEF_RM_FB_PARAM = uef_parameters.get('rm_fb_size')
    # UEF_SIM_PARAM = uef_parameters.get('re_rank_list_size')

    env = config.get('environment')

    executables = env.get('executables')
    TREC_EVAL = executables.get('trec_eval')
    RBP_EVAL = executables.get('rbp_eval')

    env_paths = env.get('paths')
    _root_dir = env_paths.get('root_dir')
    if _root_dir is None:
        _root_dir = os.getcwd()
    _root_dir = ensure_dir(_root_dir, False)
    INDEX_DIR = env_paths.get('text_index_dir')
    CIFF_INDEX = env_paths.get('ciff_index_file')
    TERRIER_INDEX = env_paths.get('terrier_index_dir')

    BATCH_NAME = env_paths.get('batch_name', '')

    assert sum((bool(TERRIER_INDEX), bool(CIFF_INDEX), bool(INDEX_DIR))) <= 1, \
        f"Only one type of Index can be specified in the configurations file"

    RESULTS_DIR = ensure_dir(os.path.join(_root_dir, env_paths.get('results_dir')), True)

    log_file = env_paths.get('log_file')
    if log_file:
        log_file = os.path.join(RESULTS_DIR, log_file)
    logging.basicConfig(filename=log_file, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging_level)
    logger = logging.getLogger(__name__)

    DB_DIR = ensure_dir(os.path.join(_root_dir, env_paths.get('db_dir')), True)

    if INDEX_DIR:
        try:
            # Index dump paths
            INDEX_DIR = ensure_dir(os.path.join(_root_dir, INDEX_DIR), create_if_not=False)
            TEXT_INV, DICT_TXT, DOC_LENS, DOC_NAMES, INDEX_GLOBALS = set_index_dump_paths(INDEX_DIR)
        except FileNotFoundError as err:
            logger.warning(err)
            logging.warning(f"The setting 'text_index_dir={INDEX_DIR}' in the config file was skipped")
            INDEX_DIR = None
    elif CIFF_INDEX:
        try:
            CIFF_INDEX = ensure_file(os.path.join(_root_dir, CIFF_INDEX))
        except FileNotFoundError as err:
            logger.warning(err)
            logger.warning(f"The setting 'ciff_index_file={CIFF_INDEX}' in the config file was skipped")
            CIFF_INDEX = None
    elif TERRIER_INDEX:
        try:
            # Index dump paths
            TERRIER_INDEX = ensure_dir(os.path.join(_root_dir, TERRIER_INDEX), create_if_not=False)
            ensure_file(os.path.join(TERRIER_INDEX, 'data.properties'))
        except FileNotFoundError as err:
            logger.warning(err)
            logging.warning(f"The setting 'terrier_index_dir={TERRIER_INDEX}'"
                            f"in the config file was skipped, data.properties file is missing")
            INDEX_DIR = None

    TEXT_QUERIES = env_paths.get('text_queries_file')
    CIFF_QUERIES = env_paths.get('ciff_queries_file')
    JSONL_QUERIES = env_paths.get('jsonl_queries_file')
    QREL_FILE = os.path.join(_root_dir, env_paths.get('qrel_file'))

    assert sum((bool(TEXT_QUERIES), bool(CIFF_QUERIES), bool(JSONL_QUERIES))) == 1, \
        f"Only one type of queries file can be specified in the configurations file"

    if TEXT_QUERIES:
        try:
            TEXT_QUERIES = ensure_file(os.path.join(_root_dir, TEXT_QUERIES))
        except FileNotFoundError as err:
            logger.warning(err)
            logger.warning(f"The setting 'text_queries_file={TEXT_QUERIES}' in the config file was skipped")
            TEXT_QUERIES = None
    elif CIFF_QUERIES:
        try:
            CIFF_QUERIES = ensure_file(os.path.join(_root_dir, CIFF_QUERIES))
        except FileNotFoundError as err:
            logger.warning(err)
            logger.warning(f"The setting 'ciff_queries_file={CIFF_QUERIES}' in the config file was skipped")
            CIFF_QUERIES = None
    elif JSONL_QUERIES:
        try:
            JSONL_QUERIES = ensure_file(os.path.join(_root_dir, JSONL_QUERIES))
        except FileNotFoundError as err:
            logger.warning(err)
            logger.warning(f"The setting 'jsonl_queries_file={JSONL_QUERIES}' in the config file was skipped")
            JSONL_QUERIES = None

    @staticmethod
    def get_logger():
        return Config.logger


# Special project types

# TODO: add a parser to the types that can receive a str object and parse it to the relevant fields
class Posting(NamedTuple):
    doc_id: int
    tf: int


class TermPosting(NamedTuple):
    term: str
    cf: int
    df: int
    posting_list: Tuple[Posting]


class TermRecord(NamedTuple):
    term: str
    id: int
    cf: int
    df: int


class TermFrequency(NamedTuple):
    term: str
    doc_id: int
    tf: int


class ResultPair(NamedTuple):
    doc_id: str
    score: float


class DocRecord(NamedTuple):
    doc_id: int
    collection_doc_id: str
    doc_len: int
