<!-- 
TODO @dangotbanned: May need to solve type parameters rendering with an extension
https://github.com/mkdocstrings/griffe/issues/405#issuecomment-4698925296
-->


::: narwhals._plan.plugins
    options:
      group_by_category: false
      inherited_members: true
      members_order: [__all__, source]
      members:
        - Plugin
        - Builtin
        - manager
        - load_plugin


::: narwhals._plan.plugins._manager.PluginManager
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - plugin
        - builtin
        - dataframe
        - lazyframe
        - series
        - evaluator
        - import_modules
        - known
