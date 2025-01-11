from __future__ import annotations

EXAMPLES = {
    "len_chars": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"words": ["foo", "Café", "345", "東京", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_len_chars(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         words_len=nw.col("words").str.len_chars()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_len_chars`:

            >>> agnostic_str_len_chars(df_pd)
              words  words_len
            0   foo        3.0
            1  Café        4.0
            2   345        3.0
            3    東京        2.0
            4  None        NaN

            >>> agnostic_str_len_chars(df_pl)
            shape: (5, 2)
            ┌───────┬───────────┐
            │ words ┆ words_len │
            │ ---   ┆ ---       │
            │ str   ┆ u32       │
            ╞═══════╪═══════════╡
            │ foo   ┆ 3         │
            │ Café  ┆ 4         │
            │ 345   ┆ 3         │
            │ 東京  ┆ 2         │
            │ null  ┆ null      │
            └───────┴───────────┘

            >>> agnostic_str_len_chars(df_pa)
            pyarrow.Table
            words: string
            words_len: int32
            ----
            words: [["foo","Café","345","東京",null]]
            words_len: [[3,4,3,2,null]]
        """,
    "replace": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"foo": ["123abc", "abc abc123"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_replace(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     df = df.with_columns(replaced=nw.col("foo").str.replace("abc", ""))
            ...     return df.to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_replace`:

            >>> agnostic_str_replace(df_pd)
                      foo replaced
            0      123abc      123
            1  abc abc123   abc123

            >>> agnostic_str_replace(df_pl)
            shape: (2, 2)
            ┌────────────┬──────────┐
            │ foo        ┆ replaced │
            │ ---        ┆ ---      │
            │ str        ┆ str      │
            ╞════════════╪══════════╡
            │ 123abc     ┆ 123      │
            │ abc abc123 ┆  abc123  │
            └────────────┴──────────┘

            >>> agnostic_str_replace(df_pa)
            pyarrow.Table
            foo: string
            replaced: string
            ----
            foo: [["123abc","abc abc123"]]
            replaced: [["123"," abc123"]]
        """,
    "replace_all": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"foo": ["123abc", "abc abc123"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_replace_all(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     df = df.with_columns(replaced=nw.col("foo").str.replace_all("abc", ""))
            ...     return df.to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_replace_all`:

            >>> agnostic_str_replace_all(df_pd)
                      foo replaced
            0      123abc      123
            1  abc abc123      123

            >>> agnostic_str_replace_all(df_pl)
            shape: (2, 2)
            ┌────────────┬──────────┐
            │ foo        ┆ replaced │
            │ ---        ┆ ---      │
            │ str        ┆ str      │
            ╞════════════╪══════════╡
            │ 123abc     ┆ 123      │
            │ abc abc123 ┆  123     │
            └────────────┴──────────┘

            >>> agnostic_str_replace_all(df_pa)
            pyarrow.Table
            foo: string
            replaced: string
            ----
            foo: [["123abc","abc abc123"]]
            replaced: [["123"," 123"]]
        """,
    "strip_chars": """
            >>> from typing import Any
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrame
            >>>
            >>> data = {"fruits": ["apple", "\\nmango"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_strip_chars(df_native: IntoFrame) -> dict[str, Any]:
            ...     df = nw.from_native(df_native)
            ...     df = df.with_columns(stripped=nw.col("fruits").str.strip_chars())
            ...     return df.to_dict(as_series=False)

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_strip_chars`:

            >>> agnostic_str_strip_chars(df_pd)
            {'fruits': ['apple', '\\nmango'], 'stripped': ['apple', 'mango']}

            >>> agnostic_str_strip_chars(df_pl)
            {'fruits': ['apple', '\\nmango'], 'stripped': ['apple', 'mango']}

            >>> agnostic_str_strip_chars(df_pa)
            {'fruits': ['apple', '\\nmango'], 'stripped': ['apple', 'mango']}
        """,
    "starts_with": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"fruits": ["apple", "mango", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_starts_with(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         has_prefix=nw.col("fruits").str.starts_with("app")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_starts_with`:

            >>> agnostic_str_starts_with(df_pd)
              fruits has_prefix
            0  apple       True
            1  mango      False
            2   None       None

            >>> agnostic_str_starts_with(df_pl)
            shape: (3, 2)
            ┌────────┬────────────┐
            │ fruits ┆ has_prefix │
            │ ---    ┆ ---        │
            │ str    ┆ bool       │
            ╞════════╪════════════╡
            │ apple  ┆ true       │
            │ mango  ┆ false      │
            │ null   ┆ null       │
            └────────┴────────────┘

            >>> agnostic_str_starts_with(df_pa)
            pyarrow.Table
            fruits: string
            has_prefix: bool
            ----
            fruits: [["apple","mango",null]]
            has_prefix: [[true,false,null]]
        """,
    "ends_with": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"fruits": ["apple", "mango", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_ends_with(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         has_suffix=nw.col("fruits").str.ends_with("ngo")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_ends_with`:

            >>> agnostic_str_ends_with(df_pd)
              fruits has_suffix
            0  apple      False
            1  mango       True
            2   None       None

            >>> agnostic_str_ends_with(df_pl)
            shape: (3, 2)
            ┌────────┬────────────┐
            │ fruits ┆ has_suffix │
            │ ---    ┆ ---        │
            │ str    ┆ bool       │
            ╞════════╪════════════╡
            │ apple  ┆ false      │
            │ mango  ┆ true       │
            │ null   ┆ null       │
            └────────┴────────────┘

            >>> agnostic_str_ends_with(df_pa)
            pyarrow.Table
            fruits: string
            has_suffix: bool
            ----
            fruits: [["apple","mango",null]]
            has_suffix: [[false,true,null]]
        """,
    "contains": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"pets": ["cat", "dog", "rabbit and parrot", "dove", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_contains(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         default_match=nw.col("pets").str.contains("parrot|Dove"),
            ...         case_insensitive_match=nw.col("pets").str.contains("(?i)parrot|Dove"),
            ...         literal_match=nw.col("pets").str.contains(
            ...             "parrot|Dove", literal=True
            ...         ),
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_contains`:

            >>> agnostic_str_contains(df_pd)
                            pets default_match case_insensitive_match literal_match
            0                cat         False                  False         False
            1                dog         False                  False         False
            2  rabbit and parrot          True                   True         False
            3               dove         False                   True         False
            4               None          None                   None          None

            >>> agnostic_str_contains(df_pl)
            shape: (5, 4)
            ┌───────────────────┬───────────────┬────────────────────────┬───────────────┐
            │ pets              ┆ default_match ┆ case_insensitive_match ┆ literal_match │
            │ ---               ┆ ---           ┆ ---                    ┆ ---           │
            │ str               ┆ bool          ┆ bool                   ┆ bool          │
            ╞═══════════════════╪═══════════════╪════════════════════════╪═══════════════╡
            │ cat               ┆ false         ┆ false                  ┆ false         │
            │ dog               ┆ false         ┆ false                  ┆ false         │
            │ rabbit and parrot ┆ true          ┆ true                   ┆ false         │
            │ dove              ┆ false         ┆ true                   ┆ false         │
            │ null              ┆ null          ┆ null                   ┆ null          │
            └───────────────────┴───────────────┴────────────────────────┴───────────────┘

            >>> agnostic_str_contains(df_pa)
            pyarrow.Table
            pets: string
            default_match: bool
            case_insensitive_match: bool
            literal_match: bool
            ----
            pets: [["cat","dog","rabbit and parrot","dove",null]]
            default_match: [[false,false,true,false,null]]
            case_insensitive_match: [[false,false,true,true,null]]
            literal_match: [[false,false,false,false,null]]
        """,
    "slice": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"s": ["pear", None, "papaya", "dragonfruit"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_slice(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         s_sliced=nw.col("s").str.slice(4, length=3)
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_slice`:

            >>> agnostic_str_slice(df_pd)  # doctest: +NORMALIZE_WHITESPACE
                         s s_sliced
            0         pear
            1         None     None
            2       papaya       ya
            3  dragonfruit      onf

            >>> agnostic_str_slice(df_pl)
            shape: (4, 2)
            ┌─────────────┬──────────┐
            │ s           ┆ s_sliced │
            │ ---         ┆ ---      │
            │ str         ┆ str      │
            ╞═════════════╪══════════╡
            │ pear        ┆          │
            │ null        ┆ null     │
            │ papaya      ┆ ya       │
            │ dragonfruit ┆ onf      │
            └─────────────┴──────────┘

            >>> agnostic_str_slice(df_pa)
            pyarrow.Table
            s: string
            s_sliced: string
            ----
            s: [["pear",null,"papaya","dragonfruit"]]
            s_sliced: [["",null,"ya","onf"]]

            Using negative indexes:

            >>> def agnostic_str_slice_negative(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(s_sliced=nw.col("s").str.slice(-3)).to_native()

            >>> agnostic_str_slice_negative(df_pd)
                         s s_sliced
            0         pear      ear
            1         None     None
            2       papaya      aya
            3  dragonfruit      uit

            >>> agnostic_str_slice_negative(df_pl)
            shape: (4, 2)
            ┌─────────────┬──────────┐
            │ s           ┆ s_sliced │
            │ ---         ┆ ---      │
            │ str         ┆ str      │
            ╞═════════════╪══════════╡
            │ pear        ┆ ear      │
            │ null        ┆ null     │
            │ papaya      ┆ aya      │
            │ dragonfruit ┆ uit      │
            └─────────────┴──────────┘

            >>> agnostic_str_slice_negative(df_pa)
            pyarrow.Table
            s: string
            s_sliced: string
            ----
            s: [["pear",null,"papaya","dragonfruit"]]
            s_sliced: [["ear",null,"aya","uit"]]
        """,
    "head": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"lyrics": ["Atatata", "taata", "taatatata", "zukkyun"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_head(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         lyrics_head=nw.col("lyrics").str.head()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_head`:

            >>> agnostic_str_head(df_pd)
                  lyrics lyrics_head
            0    Atatata       Atata
            1      taata       taata
            2  taatatata       taata
            3    zukkyun       zukky

            >>> agnostic_str_head(df_pl)
            shape: (4, 2)
            ┌───────────┬─────────────┐
            │ lyrics    ┆ lyrics_head │
            │ ---       ┆ ---         │
            │ str       ┆ str         │
            ╞═══════════╪═════════════╡
            │ Atatata   ┆ Atata       │
            │ taata     ┆ taata       │
            │ taatatata ┆ taata       │
            │ zukkyun   ┆ zukky       │
            └───────────┴─────────────┘

            >>> agnostic_str_head(df_pa)
            pyarrow.Table
            lyrics: string
            lyrics_head: string
            ----
            lyrics: [["Atatata","taata","taatatata","zukkyun"]]
            lyrics_head: [["Atata","taata","taata","zukky"]]
        """,
    "tail": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"lyrics": ["Atatata", "taata", "taatatata", "zukkyun"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_tail(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         lyrics_tail=nw.col("lyrics").str.tail()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_tail`:

            >>> agnostic_str_tail(df_pd)
                  lyrics lyrics_tail
            0    Atatata       atata
            1      taata       taata
            2  taatatata       atata
            3    zukkyun       kkyun

            >>> agnostic_str_tail(df_pl)
            shape: (4, 2)
            ┌───────────┬─────────────┐
            │ lyrics    ┆ lyrics_tail │
            │ ---       ┆ ---         │
            │ str       ┆ str         │
            ╞═══════════╪═════════════╡
            │ Atatata   ┆ atata       │
            │ taata     ┆ taata       │
            │ taatatata ┆ atata       │
            │ zukkyun   ┆ kkyun       │
            └───────────┴─────────────┘

            >>> agnostic_str_tail(df_pa)
            pyarrow.Table
            lyrics: string
            lyrics_tail: string
            ----
            lyrics: [["Atatata","taata","taatatata","zukkyun"]]
            lyrics_tail: [["atata","taata","atata","kkyun"]]
        """,
    "to_datetime": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = ["2020-01-01", "2020-01-02"]
            >>> df_pd = pd.DataFrame({"a": data})
            >>> df_pl = pl.DataFrame({"a": data})
            >>> df_pa = pa.table({"a": data})

            We define a dataframe-agnostic function:

            >>> def agnostic_str_to_datetime(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").str.to_datetime(format="%Y-%m-%d")
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_to_datetime`:

            >>> agnostic_str_to_datetime(df_pd)
                       a
            0 2020-01-01
            1 2020-01-02

            >>> agnostic_str_to_datetime(df_pl)
            shape: (2, 1)
            ┌─────────────────────┐
            │ a                   │
            │ ---                 │
            │ datetime[μs]        │
            ╞═════════════════════╡
            │ 2020-01-01 00:00:00 │
            │ 2020-01-02 00:00:00 │
            └─────────────────────┘

            >>> agnostic_str_to_datetime(df_pa)
            pyarrow.Table
            a: timestamp[us]
            ----
            a: [[2020-01-01 00:00:00.000000,2020-01-02 00:00:00.000000]]
        """,
    "to_uppercase": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"fruits": ["apple", "mango", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_to_uppercase(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         upper_col=nw.col("fruits").str.to_uppercase()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_to_uppercase`:

            >>> agnostic_str_to_uppercase(df_pd)
              fruits upper_col
            0  apple     APPLE
            1  mango     MANGO
            2   None      None

            >>> agnostic_str_to_uppercase(df_pl)
            shape: (3, 2)
            ┌────────┬───────────┐
            │ fruits ┆ upper_col │
            │ ---    ┆ ---       │
            │ str    ┆ str       │
            ╞════════╪═══════════╡
            │ apple  ┆ APPLE     │
            │ mango  ┆ MANGO     │
            │ null   ┆ null      │
            └────────┴───────────┘

            >>> agnostic_str_to_uppercase(df_pa)
            pyarrow.Table
            fruits: string
            upper_col: string
            ----
            fruits: [["apple","mango",null]]
            upper_col: [["APPLE","MANGO",null]]
        """,
    "to_lowercase": """
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"fruits": ["APPLE", "MANGO", None]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> df_pa = pa.table(data)

            We define a dataframe-agnostic function:

            >>> def agnostic_str_to_lowercase(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.with_columns(
            ...         lower_col=nw.col("fruits").str.to_lowercase()
            ...     ).to_native()

            We can then pass any supported library such as pandas, Polars, or
            PyArrow to `agnostic_str_to_lowercase`:

            >>> agnostic_str_to_lowercase(df_pd)
              fruits lower_col
            0  APPLE     apple
            1  MANGO     mango
            2   None      None

            >>> agnostic_str_to_lowercase(df_pl)
            shape: (3, 2)
            ┌────────┬───────────┐
            │ fruits ┆ lower_col │
            │ ---    ┆ ---       │
            │ str    ┆ str       │
            ╞════════╪═══════════╡
            │ APPLE  ┆ apple     │
            │ MANGO  ┆ mango     │
            │ null   ┆ null      │
            └────────┴───────────┘

            >>> agnostic_str_to_lowercase(df_pa)
            pyarrow.Table
            fruits: string
            lower_col: string
            ----
            fruits: [["APPLE","MANGO",null]]
            lower_col: [["apple","mango",null]]
        """,
}
