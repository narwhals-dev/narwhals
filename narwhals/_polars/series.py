from narwhals.dependencies import get_polars

PL = get_polars()

class PolarsSeries(PL.Series):
    ...

    def __narwhals_series__(self):
        return self
