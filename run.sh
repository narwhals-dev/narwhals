git checkout perf/duckdb-diagonal-concat
uv run --no-cache --script _benchmarks/duckdb_diagonal_concat.py -o runtime-branch.csv

git checkout main && git pull
uv run --no-cache --script _benchmarks/duckdb_diagonal_concat.py -o runtime-main.csv

uv run --no-cache --script _benchmarks/duckdb_diagonal_concat_evaluate.py
