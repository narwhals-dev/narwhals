from __future__ import annotations

import setuptools

setuptools.setup(
    name="daft-plugin",
    entry_points={
        "daft-plugin.extension": [
            "X1 = daft-plugin:ExampleOne"
            # "X2 = daft-plugin:ExampleTwo",
        ]
    },
)
