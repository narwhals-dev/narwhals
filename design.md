# Design

Let's do this differently.

Here's what I'd like to get to:

import narwhals as nw
from narwhals.translate import (
    translate_frame,
    translate_series,
    to_native,
)

dfpd = ...
df = nw.DataFrame(df_any)

df = df.with_columns(c = nw.col('a') + nw.col('b'))

result = to_native(df)

---

we need to just have a single class. can't have all this nonsense...

then, we don't even need a spec...

we can still define entrypoints though?

---

where should extract native happen?
