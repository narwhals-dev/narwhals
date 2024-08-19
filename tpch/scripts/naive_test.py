from statistics import mean
from typing import Any


def test_mean_performance(benchmark: Any) -> None:
    # Precompute some data useful for the benchmark but that should not be
    # included in the benchmark time
    data = [1, 2, 3, 4, 5]

    # Benchmark the execution of the function
    benchmark(lambda: mean(data))
