import numpy as np
import pandas as pd

import narwhals as nw


def test_with_columns_int_col_name_pandas() -> None:
    np_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = pd.DataFrame(np_matrix)
    nw_df = nw.from_native(df, eager_only=True)
    result = nw_df.with_columns(nw_df.get_column(1).alias(4)).pipe(nw.to_native)  # type: ignore[arg-type]
    expected = pd.DataFrame({0: [1, 4, 7], 1: [2, 5, 8], 2: [3, 6, 9], 4: [2, 5, 8]})
    pd.testing.assert_frame_equal(result, expected)
