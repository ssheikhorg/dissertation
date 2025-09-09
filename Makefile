install:
	uv pip install -U -r requirements.txt

fmt:
	ruff format . && ruff check . --fix

layer:
	mkdir -p lib/python && \
	pip install \
	--platform manylinux2014_x86_64 \
	--target=lib/python \
	--implementation cp \
	--python-version 3.12 \
	--only-binary=:all: --upgrade \
	-r requirements.txt &&\
	(cd lib && zip -r ../dependencies-layer.zip python) && \
	rm -rf lib
