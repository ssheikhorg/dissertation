install:
	pip install -U -r requirements.txt

fmt:
	ruff format . && ruff check . --fix

#hf_token=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