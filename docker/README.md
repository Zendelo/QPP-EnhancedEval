# Docker image with qpptk

Build the docker image via this command:

```
docker build -t qpptk .. -f ./Dockerfile
```

As soon as we have finalized everything and we have published the docker image, the build command above can be removed because others can directly use the published image.

If the image was build, you can run qpptk via Docker with the following command (e.g., printing the help message):

```
docker run --rm -ti qpptk --help
```

# Run it on full-rank approaches

Assuming you already have a pyterrier index (see below how to build it), you can run:

```
tira-run \
	--input-directory ${PWD}/sample-input-full-rank \
	--input-run ${PWD}/pyterrier-index \
	--output-directory ${PWD}/qpptk-predictions-full-rank \
	--image qpptk \
	--command '/qpptk-dummy-full-rank.sh $inputDataset $inputRun $outputDir'
```

Build the pyterrier index with this command (you can skip this step, the small test index is already added to the repository):

```
tira-run \
	--input-directory ${PWD}/sample-input-full-rank \
	--output-directory ${PWD}/pyterrier-index \
	--image webis/tira-ir-starter-pyterrier:0.0.2-base \
	--command '/workspace/pyterrier_cli.py --input $inputDataset --output $outputDir --index_directory $outputDir'
```


# Run on re-rank approaches

Assuming you have `tira` installed (use `pip3 install tira`), you can run it via:

```
tira-run \
	--input-directory ${PWD}/sample-input-re-rank \
	--output-directory ${PWD}/qpptk-predictions-re-rank \
	--image qpptk \
	--command '/qpptk-dummy-re-rank.sh $inputDataset $outputDir'
```

