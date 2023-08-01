# We start from an image that already has Terrier + PyTerrier as dependencies installed
FROM webis/tira-ir-starter-pyterrier:0.0.2-base

COPY code/qpptk/ /qpptk

WORKDIR /qpptk

RUN pip install -r requirements.txt \
	&& pip install -e .

ENTRYPOINT [ "python3", "/qpptk/qpptk/qpptk_main.py" ]
