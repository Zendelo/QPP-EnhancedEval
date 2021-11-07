import linecache
import os
from bisect import bisect_left

import matplotlib.pyplot as plt
import msgpack
import pandas as pd
from google.protobuf.internal.decoder import _DecodeVarint32
from sklearn.metrics import roc_curve, auc


def get_file_len(file_path):
    """Opens a file and counts the number of lines in it"""
    return sum(1 for _ in open(file_path))


def read_line(file_path, n):
    """Return a specific line n from a file, if the line doesn't exist, returns an empty string"""
    return linecache.getline(file_path, n)


def binary_search(list_, target):
    """Return the index of first value equal to target, if non found will raise a ValueError"""
    i = bisect_left(list_, target)
    if i != len(list_) and list_[i] == target:
        return i
    raise ValueError


def ensure_file(file):
    """Ensure a single file exists, returns the absolute path of the file if True or raises FileNotFoundError if not"""
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file))
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} doesn't exist")
    return file_path


def ensure_dir(file_path, create_if_not=True):
    """
    The function ensures the dir exists,
    if it doesn't it creates it and returns the path or raises FileNotFoundError
    """
    # tilde expansion
    file_path = os.path.normpath(os.path.expanduser(file_path))
    if os.path.isfile(file_path):
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        if create_if_not:
            try:
                os.makedirs(directory)
            except FileExistsError:
                # This exception was added for multiprocessing, in case multiple process try to create the directory
                pass
        else:
            raise FileNotFoundError(f"The directory {directory} doesnt exist, create it or pass create_if_not=True")
    return directory


def read_message(buffer, n, message_type):
    """
    The function used to read and parse a protobuf message, specifically for delimited protobuf files
    """
    message = message_type()
    msg_len, new_pos = _DecodeVarint32(buffer, n)
    n = new_pos
    msg_buf = buffer[n:n + msg_len]
    n += msg_len
    message.ParseFromString(msg_buf)
    return n, message


def transform_list_to_counts_dict(_list):
    counts = [_list.count(i) for i in _list]
    return {i: j for i, j in zip(_list, counts)}


def jaccard_similarity(set_1, set_2):
    return len(set_1.intersection(set_2)) / len(set_1.union(set_2))


def overlap_coefficient(set_1, set_2):
    return len(set_1.intersection(set_2)) / min(len(set_1), len(set_2))


def duplicate_qrel_file_to_qids(qrel_file, qids):
    qrels_df = pd.read_table(qrel_file, delim_whitespace=True, names=['qid', 'iter', 'doc', 'rel'], dtype=str)
    topics_dict = {qid: qid.split('-', 1)[0] for qid in qids}  # {qid:topic}
    result = []
    for qid, topic in topics_dict.items():
        result.append(
            qrels_df.loc[qrels_df['qid'] == topic,
                         ['iter', 'doc', 'rel']].assign(qid=qid).loc[:, ['qid', 'iter', 'doc', 'rel']])
    res_df = pd.concat(result, axis=0)
    res_df.to_csv(qrel_file.replace('.qrels', '_mod.qrels'), sep=' ', index=False, header=False)


def add_topic_to_qdf(qdf: pd.DataFrame):
    """This function used to add a topic column to a queries DF"""
    columns = qdf.columns.to_list()
    if 'topic' not in columns:
        if 'qid' in columns:
            qdf = qdf.assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
        else:
            qdf = qdf.reset_index().assign(topic=lambda x: x.qid.apply(lambda y: y.split('-')[0]))
    columns = qdf.columns.to_list()
    return qdf.loc[:, columns[-1:] + columns[:-1]]


def msgpack_encode(vector):
    return msgpack.packb(vector)


def msgpack_decode(serialized_vector):
    return msgpack.unpackb(serialized_vector)


def read_trec_res_file(file_name):
    """
    Assuming data is in trec format results file with 6 columns, 'Qid entropy cross_entropy Score
    '"""
    data_df = pd.read_csv(file_name, delim_whitespace=True, header=None, index_col=0,
                          names=['qid', 'Q0', 'docNo', 'docRank', 'docScore', 'ind'],
                          dtype={'qid': str, 'Q0': str, 'docNo': str, 'docRank': int, 'docScore': float,
                                 'ind': str})
    data_df = data_df.filter(['qid', 'docNo', 'docRank', 'docScore'], axis=1)
    data_df.index = data_df.index.astype(str)
    data_df.sort_values(by=['qid', 'docRank'], ascending=True, inplace=True)
    return data_df


def plot_roc(y_test, y_pred, predictor_name):
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    plt.title(f'ROC {predictor_name}')
    plt.plot(fpr, tpr, 'b', label=f'ROC (AUC = {auc(fpr, tpr):0.2f})')
    # plt.plot(fpr, threshold, label='Threshold')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate (Sensitivity/Recall)')
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.show()
