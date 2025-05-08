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
    # install duckdb nightly so mypy recognises duckdb.SQLExpression
	$(VENV_BIN)/uv pip install -U --pre duckdb
	$(VENV_BIN)/uv pip install -e . --group typing
	$(VENV_BIN)/mypy
