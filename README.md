# QPP-EnhancedEval
Code to Reproduce ECIR 2021 paper "An Enhanced Evaluation Framework for Query Performance Prediction"


## Run it locally with TIRA

Please ensure that you have python >= 3.7, Docker, and tira-run installed (`pip3 install tira`).

```
tira-run \
	--input-dataset workshop-on-open-web-search/retrieval-20231027-training \
	--approach ir-benchmarks/qpptk/all-predictors \
	--input-run 'ir-benchmarks/tira-ir-starter/Index (tira-ir-starter-pyterrier)'
```

In this command, `--input-dataset` refers to a dataset identifier (can be replaced with `--input-directory` to run on data not available in ir_datasets), `--approach` is the identifier of the qpptk software in TIRA that did run all available predictors, and `--input-run` specifies that a PyTerrier index should be constructed / passed produced by the software with the corresponding id.

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

