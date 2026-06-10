<!-- NOTE: A bunch of stuff with varying levels of docs-->

<!-- TODO @dangotbanned: Really need to organize these better so they can be referenced from
somewhere less private
-->

::: narwhals._plan._meta
    options:
      members_order: source
      members:
        - SlottedMeta
        - ImmutableMeta
        - ExprIRMeta

::: narwhals._plan._immutable.Immutable
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - __immutable_keys__
        - __immutable_values__
        - __immutable_items__
        - __immutable_hash__
        - __copy__
        - __deepcopy__
        - __eq__
        - __hash__
        - __init__
        -  __replace__
        - __str__
        - to_dict

::: narwhals._plan._nodes
    options:
      members_order: source
      members:
        - node
        - nodes
        - Node
        - ExprNode
        - SingleExpr
        - MultipleExpr
        - ExprTraverser

::: narwhals._plan._flags
    options:
      members:
        - FunctionFlags

::: narwhals._plan._dispatch
    options:
      members_order: source
      members:
          - ConstructorDispatch
          - Dispatch
          - DispatcherOptions
          - ExprIRDispatch
          - FunctionDispatch
          - FunctionExprDispatch
          - NoDispatch
          - get_dispatch_name


::: narwhals._plan._dtype
    options:
      members:
          - ResolveDType
          - _FunctionAccessor
          - _ExprIRAccessor
          - GetDType
          - JustDType
          - ExprIRSameDType
          - ExprIRMapFirst
          - ExprIRVisitor
          - FunctionVisitor
          - FunctionSameDType
          - FunctionMapFirst
          - FunctionMapAll
          - IntoResolveDType
          - Visitor


::: narwhals._plan._expansion
    options:
      members:
        - prepare_projection
        - expand_selectors
        - parse_expand_selectors
        - Expander

::: narwhals._plan._parameters
    options:
      members:
        - Parameters
        - Unary
        - Binary
        - Ternary
        - Variadic
        - Constraint
        - Arity


::: narwhals._plan.common
    options:
      members:
        - temp
        - closed_kwds


::: narwhals._plan.options
    options:
      members:
        - SortOptions
        - SortMultipleOptions
        - RankOptions
        - EWMOptions
        - RollingOptions
        - RollingVarOptions
        - ExplodeOptions
        - UniqueOptions
        - VConcatOptions
        - JoinOptions
        - JoinAsofBy
        - JoinAsofOptions
        - UnpivotOptions

::: narwhals._plan.schema
    options:
      members:
        - FrozenSchema
        - IntoSchema
        - IntoFrozenSchema
        - HasSchema


::: narwhals._plan.plugins
    options:
      members:
        - Builtin
        - Plugin
        - manager
        - load_plugin

::: narwhals._plan.plugins._manager
    options:
      members:
        - PluginManager