from narwhals import dtypes


class ArrowNamespace:
    Int64 = dtypes.Int64
    Int32 = dtypes.Int32
    Int16 = dtypes.Int16
    Int8 = dtypes.Int8
    UInt64 = dtypes.UInt64
    UInt32 = dtypes.UInt32
    UInt16 = dtypes.UInt16
    UInt8 = dtypes.UInt8
    Float64 = dtypes.Float64
    Float32 = dtypes.Float32
    Boolean = dtypes.Boolean
    Object = dtypes.Object
    Categorical = dtypes.Categorical
    String = dtypes.String
    Datetime = dtypes.Datetime
    Date = dtypes.Date

    # --- not in spec ---
    def __init__(self) -> None: ...
