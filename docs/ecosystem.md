---
render_macros: true
---

{% from "cards.html" import project_card %}

# Ecosystem

The Narwhals Ecosystem is comprised of tools that rely on us and the tabular data packages we support either directly or through 
the [plugins system](/narwhals/extending/)

## Narwhals is Used by

The following is a non-exhaustive list of {{ projects.downstream | length }} libraries and tools that choose to use Narwhals
for their dataframe interoperability needs:

<div class="project-grid">

{% for project in projects.downstream %}
    {{ project_card(project.name, project.homepage, project.logo) }}
{% endfor %}

</div>

If your project is missing from the list, feel free to open a PR to add it.

If you would like to chat with us, or if you need any support, please [join our Discord server](https://discord.gg/V3PqtB4VA4).

## Directly Supported Tabular Data Packages

Narwhals directly provides support for {{ projects.upstream | length }} tabular data packages.

<div class="project-grid">

{% for project in projects.upstream %}
    {{ project_card(project.name, project.homepage, project.logo) }}
{% endfor %}

</div>

## Plugin Supported Tabular Data Packages

Narwhals additionally provides support for {{ projects.plugin | length }} tabular data package(s), through plugins.
See [Extensions and Plugins](/narwhals/extending/) for how to write your own plugin and bring Narwhals to another tabular data package.

<div class="project-grid">

{% for project in projects.plugin %}
    {{ project_card(project.name, project.homepage, project.logo) }}
{% endfor %}

</div>

## Other Related projects

### Array API

Array counterpart to the DataFrame API, see [here](https://data-apis.org/array-api/2022.12/index.html).

### PyCapsule Interface

Allows C extension modules to safely share pointers to C data structures with Python code
and other C modules, encapsulating the pointer with a name and optional destructor to
manage resources and ensure safe access,
see [here](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html) for details.

Narwhals supports exporting a DataFrame via the Arrow PyCapsule Interface. See
[Universal dataframe support with the Arrow PyCapsule Interface + Narwhals](https://labs.quansight.org/blog/narwhals-pycapsule)
for how you can use them together.

### Ibis

Pitched as "The portable Dataframe library", Ibis provides a Pythonic frontend
to various SQL (as well as Polars LazyFrame) engines. Some differences with Narwhals are:

* Narwhals allows you to write "Dataframe X in, Dataframe X out" functions.
  Ibis allows materialising to pandas, Polars (eager), and PyArrow, but has
  no way to get back to the input type exactly (e.g. `Enum`s don't round-trip in Ibis)
* Narwhals respects input data types as much as possible, Ibis doesn't
  support Categorical / Enum.
* Narwhals separates between lazy and eager APIs, with the eager API
  providing very fine control over dataframe operations (slicing rows and
  columns, iterating over rows, getting values out of the dataframe as
  Python scalars). Ibis is more focused on lazy execution.
* Narwhals is extremely lightweight and comes with zero required dependencies,
  Ibis requires pandas and PyArrow for all backends.
* Narwhals uses a subset of the Polars API, Ibis uses its own
  pandas/dplyr-inspired API.
* Ibis currently supports more backends than Narwhals
* Narwhals supports pandas and Dask, which Ibis has deprecated support for.

Although people often ask about the two tools, we consider them to be
very different and not in competition. In particular,
**Narwhals supports ibis Tables**, meaning that dataframe-agnostic code
written using Narwhals' lazy API also supports Ibis.
