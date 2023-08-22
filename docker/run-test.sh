qpptk --pyterrier_index {PATH_TO_INDEX} --queries {PATH_TO_QUERIES.jsonl} --output_dir {PREDICTIONS_DIR} --predictors {all, pre, post, [scq, idf, var]} [--top_docs k, --retrieval_results {PATH_TO_TREC_RESULT}]

# if no top_docs run on a default list of params [5, 10, 25, 50, 75, 100, 150, 200,250]
# generate a separate file for each predictor
# results file format: {'qid': QID, 'predictor_name': MaxIDF, 'prediction_score':0.2, 'hparameters': {top_docs: 10}}
