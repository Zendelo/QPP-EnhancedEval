# QPP-EnhancedEval
Code to Reproduce ECIR 2021 paper "An Enhanced Evaluation Framework for Query Performance Prediction"


## Run it locally with TIRA

Please ensure that you have python >= 3.7, Docker, and tira-run installed (`pip3 install tira`).

```
tira-run \
	--input-directory ${PWD}/docker/sample-input-full-rank \
	--input-run ${PWD}/docker/pyterrier-index \
	--image mam10eks/qpptk:0.0.1 \
	--command 'python3 /qpptk_main.py -ti $inputRun/index/ --jsonl_queries $inputDataset/queries.jsonl --predict --retrieve --output $outputDir --cleanOutput'
```

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

