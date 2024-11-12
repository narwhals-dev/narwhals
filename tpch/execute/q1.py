from __future__ import annotations

from queries import q1

from . import IO_FUNCS
from . import lineitem

print(q1.query(IO_FUNCS["pandas[pyarrow]"](lineitem)))
print(q1.query(IO_FUNCS["polars[lazy]"](lineitem)).collect())
print(q1.query(IO_FUNCS["pyarrow"](lineitem)))
