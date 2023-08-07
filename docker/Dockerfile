# We start from an image that already has Terrier + PyTerrier as dependencies installed
FROM webis/tira-ir-starter-pyterrier:0.0.2-base

COPY code/qpptk/ /qpptk

WORKDIR /qpptk

RUN pip install -r requirements.txt \
	&& pip install -e .

#ONLY For Debug purposes
RUN apt-get install -y jq \
	&& echo '#!/bin/sh\n\necho "Read Topics from ${1}queries.jsonl"\necho "Read Pyterrier-Index from ${2}/index:"\n ls ${2}/index\nsleep 2s\ncat ${1}/queries.jsonl | jq -c '\''{"qid": .qid, "query": .query, "predictions": {"max-idf": 1}}'\'' > ${3}/predictions.jsonl' > /qpptk-dummy-full-rank.sh \
	&& chmod +x /qpptk-dummy-full-rank.sh \
	&& echo '#!/bin/sh\n\necho "Read query-document pairs from ${1}/rerank.jsonl.gz"\n\nsleep 2s\nzcat ${1}/rerank.jsonl.gz | jq -c '\''{"qid": .qid, "query": .query, "predictions": {"max-idf": 1}}'\'' > ${2}/predictions.jsonl' > /qpptk-dummy-re-rank.sh \
	&& chmod +x /qpptk-dummy-re-rank.sh

ENTRYPOINT [ "python3", "/qpptk/qpptk/qpptk_main.py" ]
