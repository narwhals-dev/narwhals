# Mostly based on polars Makefile
# https://github.com/pola-rs/polars/blob/main/py-polars/Makefile

.DEFAULT_GOAL := help

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

.PHONY: docs-serve
docs-serve: ## Build and serve the docs locally
	uv run --group docs --extra dask --extra ibis utils/generate_backend_completeness.py
	uv run --group docs utils/generate_zen_content.py
	uv run --group docs zensical serve

.PHONY: docs-clean-serve
docs-clean-serve: ## Rebuild docs from a clean state and serve them locally
	uv run --group docs zensical build --clean
	$(MAKE) docs-serve

.PHONY: run-ci
run-ci:  ## Print resolved deps, then run a command via uv. Usage: make run-ci DEPS="<groups/extras>" CMD="<command>" [RUN_ONLY="<uv-run-only flags, e.g. --isolated, --with X, --no-sync>"]
	uv export --no-annotate --no-hashes $(DEPS)
	uv run $(DEPS) $(RUN_ONLY) $(CMD)

.PHONY: show-deps
show-deps:  ## Print resolved deps Usage: make show-deps DEPS="<groups/extras>"
	uv export --no-annotate --no-hashes $(DEPS)

# 1 `uv run` isolates `sys.modules` monkeypatching from causing failures in other tests
# 2 `uv run` is the rest
# Both use `--cov-append` and should be run before `main`
.PHONY: run-ci-plan
run-ci-plan: ## Very long command that I can't keep fixing merge conflicts for in yml
	$(MAKE) show-deps DEPS="--group core-tests"
	uv run --group core-tests pytest tests/plan -m unsafe_globals --numprocesses=1 --cov=src/narwhals/_plan --cov=tests/plan --cov-fail-under=0
	uv run --group core-tests pytest tests/plan src/narwhals/_plan --cov=src/narwhals/_plan --cov=tests/plan --cov-fail-under=0 --doctest-modules --runslow --durations=30 --cov-append