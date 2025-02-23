import narwhals as nw
import pytest

@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (nw.col('a'), False),
        (nw.col('a').mean(), False),
        (nw.col('a').cum_sum(), True),
        (nw.col('a').cum_sum().over(_order_by='id'), False),
        ((nw.col('a').cum_sum()+1).over(_order_by='id'), True),
        (nw.col('a').cum_sum().cum_sum().over(_order_by='id'), True),
    ]
)
def test_has_open_windows(expr: nw.Expr, expected: bool) -> None:
    assert expr._metadata['has_open_windows'] == expected
