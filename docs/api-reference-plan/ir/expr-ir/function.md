# Function

<!-- TODO @dangotbanned: Export everything to a (mostly) flat function namespace
https://github.com/narwhals-dev/narwhals/blob/d58d05a9b1435f01d3497571f1869bf6e890b12a/src/narwhals/_plan/expressions/functions.py#L3-L5
-->

::: narwhals._plan._function
    options:
      members_order: source
      members:
        - Function
        - UnaryFunction
        - BinaryFunction
        - TernaryFunction
        - HorizontalFunction


::: narwhals._plan.expressions.functions
    options:
      members:
        - Abs
        - AsStruct
        - Ceil
        - Clip
        - ClipLower
        - ClipUpper
        - Coalesce
        - CumAgg
        - CumCount
        - CumMax
        - CumMin
        - CumProd
        - CumSum
        - Diff
        - DropNulls
        - EwmMean
        - Exp
        - FillNull
        - FillNullWithStrategy
        - Floor
        - GatherEvery
        - Hist
        - HistBinCount
        - HistBins
        - Kurtosis
        - Log
        - MapBatches
        - MaxHorizontal
        - MeanHorizontal
        - MinHorizontal
        - ModeAll
        - ModeAny
        - NullCount
        - Pow
        - Rank
        - ReplaceStrict
        - ReplaceStrictDefault
        - RollingMean
        - RollingStd
        - RollingSum
        - RollingVar
        - RollingWindow
        - Round
        - SampleFrac
        - SampleN
        - Shift
        - Skew
        - Sqrt
        - SumHorizontal
        - Unique

## Boolean

::: narwhals._plan.expressions.boolean
    options:
      members:
        - All
        - AllHorizontal
        - Any
        - AnyHorizontal
        - BooleanFunction
        - IsBetween
        - IsDuplicated
        - IsFinite
        - IsFirstDistinct
        - IsInExpr
        - IsInSeq
        - IsInSeries
        - IsLastDistinct
        - IsNan
        - IsNotNan
        - IsNotNull
        - IsNull
        - IsUnique
        - Not


## Categorical

::: narwhals._plan.expressions.categorical
    options:
      members:
        - CategoricalFunction
        - GetCategories



## List

::: narwhals._plan.expressions.lists
    options:
      members:
        - ListFunction
        - All
        - Any
        - Contains
        - First
        - Get
        - Join
        - Last
        - Len
        - Max
        - Mean
        - Median
        - Min
        - NUnique
        - Sort
        - Sum
        - Unique


## Range

::: narwhals._plan.expressions.ranges
    options:
      members:
        - RangeFunction
        - DateRange
        - IntRange
        - LinearSpace

## String

::: narwhals._plan.expressions.strings
    options:
      members:
        - StringFunction
        - ConcatStr
        - Contains
        - EndsWith
        - LenChars
        - Replace
        - ReplaceAll
        - Slice
        - Split
        - StartsWith
        - StripChars
        - ToDate
        - ToDatetime
        - ToLowercase
        - ToTitlecase
        - ToUppercase
        - ZFill

## Struct

::: narwhals._plan.expressions.struct
    options:
      members:
        - StructFunction
        - FieldByName
        

## Temporal

::: narwhals._plan.expressions.temporal
    options:
      members:
        - TemporalFunction
        - ConvertTimeZone
        - Date
        - Day
        - Hour
        - Microsecond
        - Millisecond
        - Minute
        - Month
        - Nanosecond
        - OffsetBy
        - OrdinalDay
        - ReplaceTimeZone
        - Second
        - Timestamp
        - ToString
        - TotalMicroseconds
        - TotalMilliseconds
        - TotalMinutes
        - TotalNanoseconds
        - TotalSeconds
        - Truncate
        - WeekDay
        - Year