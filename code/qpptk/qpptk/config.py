import logging
import os
from typing import NamedTuple, Tuple

import toml

from qpptk import ensure_dir, ensure_file

CONFIG_FILE = './qpptk/config.toml'

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
    logging_level = parameters.get('logging_level', 'DEBUG')
    N_PROC = parameters.get('num_processes', 1)
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S',
                        level=logging_level)
    logger = logging.getLogger(__name__)
    prediction_parameters = parameters.get('prediction')
    WIG_LIST_SIZE = prediction_parameters.get('wig_list_size')
    NQC_LIST_SIZE = prediction_parameters.get('nqc_list_size')
    SMV_LIST_SIZE = prediction_parameters.get('smv_list_size')
    CLARITY_FB_TERMS = prediction_parameters.get('clarity_fb_terms')
    CLARITY_LIST_SIZE = prediction_parameters.get('clarity_list_size')
    uef_parameters = prediction_parameters.get('uef')
    UEF_RM_FB_PARAM = uef_parameters.get('rm_fb_size')
    UEF_SIM_PARAM = uef_parameters.get('re_rank_list_size')

    env = config.get('environment')

    executables = env.get('executables')
    TREC_EVAL = executables.get('trec_eval')

    env_paths = env.get('paths')
    _root_dir = env_paths.get('root_dir')
    if _root_dir is None:
        _root_dir = os.getcwd()
    _root_dir = ensure_dir(_root_dir, False)
    INDEX_DIR = env_paths.get('text_index_dir')
    CIFF_INDEX = env_paths.get('ciff_index_file')
    assert not (INDEX_DIR and CIFF_INDEX), f"Only one type of Index can be specified in the configurations file"

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

    TEXT_QUERIES = env_paths.get('text_queries_file')
    CIFF_QUERIES = env_paths.get('ciff_queries_file')
    assert not (TEXT_QUERIES and CIFF_QUERIES), f"Only a single type of queries file can be specified" \
                                                f" in the configurations file"
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

    RESULTS_DIR = ensure_dir(os.path.join(_root_dir, env_paths.get('results_dir')), True)
    DB_DIR = ensure_dir(os.path.join(_root_dir, env_paths.get('db_dir')), True)

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
