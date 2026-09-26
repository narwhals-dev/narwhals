# Mostly based on polars Makefile
# https://github.com/pola-rs/polars/blob/main/py-polars/Makefile

.DEFAULT_GOAL := help

# coverage's execv/fork patches raise on Windows; collapse them to `subprocess`
# there (coverage dedupes) and keep the default values elsewhere.
# See `tool.coverage.run.patch` in pyproject.toml.
ifeq ($(OS),Windows_NT)
export COVERAGE_PATCH_EXECV ?= subprocess
export COVERAGE_PATCH_FORK ?= subprocess
endif

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: lint
lint: ## Run code formatting and linting via ruff
	prek run ruff-format --all-files
	prek run ruff-check --all-files

.PHONY: typing
typing: ## Run type checkers
	uv run --group typing pyright
	uv run --group typing mypy
	uv run --group typing pyrefly check

.PHONY: typing-coverage
typing-coverage: ## Run type checkers
	uv run --group typing pyrefly coverage check src/narwhals --public-only

.PHONY: docs-dynamic-content
docs-dynamic-content:  ## Regenerate the dynamic docs pages (API completeness tables, docs/this.md, ...)
	uv run --group docs --extra dask --extra ibis utils/generate_backend_completeness.py
	uv run --group docs utils/generate_zen_content.py

.PHONY: docs-build
docs-build: ## Build the docs from a clean state, failing on warnings
	$(MAKE) docs-dynamic-content
	uv run --group docs zensical build --clean --strict

.PHONY: docs-serve
docs-serve: ## Serve the docs locally
	$(MAKE) docs-dynamic-content
	uv run --group docs zensical serve

.PHONY: docs-clean-serve
docs-clean-serve: ## Rebuild docs from a clean state and serve them locally
	$(MAKE) docs-build
	uv run --group docs zensical serve

.PHONY: run-ci
run-ci:  ## Print resolved deps, then run a command via uv (no coverage; used by doctests, narrow-dep runs, tpch, ibis, modin). Usage: make run-ci DEPS="<groups/extras>" CMD="<command>" [RUN_ONLY="<uv-run-only flags, e.g. --isolated, --with X, --no-sync>"]
	uv export --no-annotate --no-hashes $(DEPS)
	uv run $(DEPS) $(RUN_ONLY) $(CMD)

.PHONY: run-ci-coverage
run-ci-coverage:  ## Like run-ci but under coverage (run -> combine -> report). Usage: make run-ci-coverage DEPS="<groups/extras>" CMD="<command, e.g. pytest tests --nw-backends=...>" [FAIL_UNDER="<int>"] [COV_SOURCE="<path to scope coverage to, e.g. src/narwhals/_spark_like; drives both 'run --source' and 'report --include'>"] [RUN_ONLY="<uv-run-only flags>"]
	uv export --no-annotate --no-hashes $(DEPS)
	uv run $(DEPS) $(RUN_ONLY) coverage run $(if $(COV_SOURCE),--source=$(COV_SOURCE)) -m $(CMD)
	uv run $(DEPS) $(RUN_ONLY) coverage combine
	uv run $(DEPS) $(RUN_ONLY) coverage report $(if $(COV_SOURCE),--include=$(COV_SOURCE)/*) $(if $(FAIL_UNDER),--fail-under=$(FAIL_UNDER))

.PHONY: doctest
doctest:  ## Run doctests
	make run-ci \
		DEPS="--extra pandas --extra dask --group core-tests --group sklearn" \
		CMD="pytest src --doctest-modules"

.PHONY: test-full-coverage
test-full-coverage:  ## Run the full test suite with 100% coverage across all constructors as in CI
	PYTEST_ADDOPTS="--numprocesses=logical" make run-ci-coverage \
		DEPS="--extra pandas --extra dask --group core-tests --group sklearn --group plugins" \
		FAIL_UNDER="100" \
		CMD="pytest tests --runslow --durations=30 --nw-backends=pandas,pandas[nullable],pandas[pyarrow],pyarrow,polars[eager],polars[lazy],dask,duckdb,sqlframe"
