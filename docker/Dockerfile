# docker build -f docker/Dockerfile -t mam10eks/qpptk:0.0.1 .
# We start from an image that already has all dependencies installed
FROM mam10eks/qpptk:0.0.2-dev

COPY code/qpptk/ /workspaces/QPP-EnhancedEval/code/qpptk

RUN pip install -e . \
	&& pytest

ENTRYPOINT [ "python3", "/qpptk_main.py" ]
