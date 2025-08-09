install:
	pip install -U -r requirements.txt

fmt:
	ruff format . && ruff check . --fix
