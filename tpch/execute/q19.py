from queries import q19

from . import IO_FUNCS
from . import lineitem
from . import part

fn = IO_FUNCS["pandas"]
print(q19.query(fn(lineitem), fn(part)))

fn = IO_FUNCS["pandas[pyarrow]"]
print(q19.query(fn(lineitem), fn(part)))

fn = IO_FUNCS["polars[eager]"]
print(q19.query(fn(lineitem), fn(part)))

fn = IO_FUNCS["polars[lazy]"]
print(q19.query(fn(lineitem), fn(part)).collect())
