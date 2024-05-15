# Roadmap

Where do we go from here? What are the project's priorities? In no particular order:

## Tests and docs coverage

Every method should have a good docstring with an example.

CI should test across a variety of pandas and Polars versions.
Currently we just test latest and minimum versions.

## API coverage

Narwhals should be complete enough to be able to execute all 22 tpc-h queries.
Currently, it can execute the first 7.

## Other backends?

Narwhals is extesible - can we make it as easy as possible for backends to become
compatible with it?

## Query optimisation

Can we insert a light layer to do some simple query optimisation for pandas?

## Use cases

We're currently investigating whether we can make scikit-lego dataframe-agnostic.

Ideas for other projects we could support? If so, please post them [here](https://github.com/narwhals-dev/narwhals/issues/62).
