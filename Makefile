# Mostly based on polars Makefile
# https://github.com/pola-rs/polars/blob/main/py-polars/Makefile

.DEFAULT_GOAL := help

SHELL=bash
VENV=./.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif


.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: typing
typing: ## Run typing checks
	$(VENV_BIN)/uv pip install \
		--upgrade \
		--editable test-plugin/. \
		--editable . \
		--group typing
	$(VENV_BIN)/uv run --no-sync pyright
	$(VENV_BIN)/uv run --no-sync mypy

.PHONY: docs-serve
docs-serve:  # Build and serve the docs locally
	$(VENV_BIN)/uv pip install \
		--upgrade \
		--editable test-plugin/. \
		--editable . \
		--group docs
	$(VENV_BIN)/uv run --no-sync utils/generate_backend_completeness.py
	$(VENV_BIN)/uv run --no-sync utils/generate_zen_content.py
	$(VENV_BIN)/uv run --no-sync zensical serve

.PHONY: test
test: ## Run unittest
	$(VENV_BIN)/uv pip install \
		--upgrade \
		--editable test-plugin/. \
		--editable .[ibis,modin,pyspark] \
		--group core \
		--group tests
	$(VENV_BIN)/uv run --no-sync coverage run -m pytest tests --all-cpu-constructors --numprocesses=logical
	$(VENV_BIN)/uv run --no-sync coverage combine
	$(VENV_BIN)/uv run --no-sync coverage report --fail-under=95
