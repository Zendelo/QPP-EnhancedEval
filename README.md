# QPP-EnhancedEval
Code to Reproduce ECIR 2021 paper "An Enhanced Evaluation Framework for Query Performance Prediction"


## Run it locally with TIRA

Please ensure that you have python >= 3.7, Docker, and tira-run installed (`pip3 install tira`).

```
tira-run \
	--input-dataset workshop-on-open-web-search/query-processing-20231027-training \
	--input-run 'workshop-on-open-web-search/tira-ir-starter/Index (tira-ir-starter-pyterrier)' \
	--image mam10eks/qpptk:0.0.1 \
	--command 'python3 /qpptk_main.py -ti $inputRun/index/ --jsonl_queries $inputDataset/queries.jsonl --predict --retrieve --output $outputDir --cleanOutput --stats_index_path /tmp'
```

  File "/workspaces/QPP-EnhancedEval/code/qpptk/qpptk/global_manager.py", line 33, in run_pre_prediction_process
    max_idf = process.calc_max_idf()
  File "/workspaces/QPP-EnhancedEval/code/qpptk/qpptk/pre_retrieval_predictors.py", line 30, in calc_max_idf
    return np.log(np.array(self.total_docs) / self.terms_df).max()
  File "/usr/local/lib/python3.10/dist-packages/numpy/core/_methods.py", line 40, in _amax
    return umr_maximum(a, axis, None, out, keepdims, initial, where)
ValueError: zero-size array to reduction operation maximum which has no identity


## Build the Docker Images

Build the docker image via:
```
docker build -f docker/Dockerfile -t mam10eks/qpptk:0.0.1 .
```

If you update any dependencies, please rebuild the dev container via:
```
docker build -f docker/Dockerfile.dev -t mam10eks/qpptk:0.0.1-dev .
```

## Upload to TIRA

```
docker tag mam10eks/qpptk:0.0.1 registry.webis.de/code-research/tira/tira-user-qpptk/qpptk:0.0.1
docker push registry.webis.de/code-research/tira/tira-user-qpptk/qpptk:0.0.1
```

