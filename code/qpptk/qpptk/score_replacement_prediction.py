from pathlib import Path
import tempfile
import pandas as pd


def replace_scores_in_run_file_with_reference_scores(run_file: Path, reference_run_file: Path) -> Path:
    """
    This class is used to replace the retrieval scores of a base model (e.g., BM25) with the scores of other neural models.

    The ranking comes from the base model, but the scores used in query used in query performance predictors like WIG come from another model with the intendion to vary them in their size, (e.g., use BM25 rankings as run that induces the ranking and monoT5 small vs base vs large vs 3b etc to induce the scores.).

    :param run_file: The run file that induces the ranking of documents (e.g., BM25).
    :param reference_run_file: The run file that induces the scores of documents (e.g., derived by monoT5 small).
    """
    from .utility_functions import read_trec_res_file
    run_file_parsed = read_trec_res_file(run_file)
    reference_scores = {}
    for qid, i in read_trec_res_file(reference_run_file).iterrows():
        reference_scores[(qid, i['docNo'])] = i['docScore']

    run_file_parsed['docScore'] = run_file_parsed.apply(lambda x: reference_scores[(x.name, x['docNo'])], axis=1)
    merged_run = []
    for qid, i in run_file_parsed.iterrows():
        merged_run += [{'qid': qid, 'Q0': '0', 'docNo': i['docNo'], 'docRank': i['docRank'], 'docScore': reference_scores[(qid, i['docNo'])], 'system': 'merged'}]

    ret = Path(tempfile.NamedTemporaryFile(delete=False).name)
    pd.DataFrame(merged_run).to_csv(ret, sep=' ', header=False, index=False)
    return ret
