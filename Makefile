.PHONY: check format

check:
	uvx ruff check --fix src
	uvx ty check src

format:
	uvx ruff check --select I --fix src
	uvx ruff format src