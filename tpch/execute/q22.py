from queries import q22

from . import IO_FUNCS
from . import customer
from . import orders

fn = IO_FUNCS["pandas"]
print(q22.query(fn(customer), fn(orders)))

fn = IO_FUNCS["pandas[pyarrow]"]
print(q22.query(fn(customer), fn(orders)))

fn = IO_FUNCS["polars[eager]"]
print(q22.query(fn(customer), fn(orders)))

fn = IO_FUNCS["polars[lazy]"]
print(q22.query(fn(customer), fn(orders)).collect())
