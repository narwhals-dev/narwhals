.DEFAULT_GOAL := help

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort

.PHONY: test
test:  ## Run the narwhals test suite against narwhals-dict (known failures excluded)
	python run_tests.py

.PHONY: test-all
test-all:  ## Run the narwhals test suite against narwhals-dict, including known failures
	cd ../.. && uv run pytest tests -p narwhals_dict.testing --use-external-constructor --tb=no

.PHONY: test-update
test-update:  ## Regenerate the known-failures list in run_tests.py
	python run_tests.py --update
