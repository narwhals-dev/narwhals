# Contributing

Thank you for your interest in contributing to Narwhals! Any kind of improvement is welcome!

## **TLDR**

If you've got experience with open source contributions, the following instructions might suffice:

- clone repo: `git clone git@github.com:narwhals-dev/narwhals.git narwhals-dev`
- `cd narwhals-dev/`
- `git remote rename origin upstream`
- `git remote add origin <your fork goes here>`
- `uv venv -p 3.12`
- `. .venv/bin/activate`
- `uv pip install -U -e . --group local-dev`
- To run tests: `pytest`
- To run all linting checks: `pre-commit run --all-files`
- To run static typing checks: `make typing`

For more detailed and beginner-friendly instructions, see below!

## Local development vs Codespaces

You can contribute to Narwhals in your local development environment, using python3, git and your editor of choice.
You can also contribute to Narwhals using [Github Codespaces](https://docs.github.com/en/codespaces/overview) - a development environment that's hosted in the cloud.
This way you can easily start to work from your browser without installing git and cloning the repo.
Scroll down for instructions on how to use [Codespaces](#working-with-codespaces).

## Working with local development environment

### 1. Make sure you have git on your machine and a GitHub account

Open your terminal and run the following command:

```bash
git --version
```

If the output looks like `git version 2.34.1` and you have a personal account on GitHub - you're good to go to the next step.
If the terminal output informs about `command not found` you need to [install git](https://docs.github.com/en/get-started/quickstart/set-up-git).

If you're new to GitHub, you'll need to create an account on [GitHub.com](https://github.com/) and verify your email address.

You should also [check for existing SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/checking-for-existing-ssh-keys) and
[generate and add a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
if you don't have one already.

### 2. Fork the repository

Go to the [main project page](https://github.com/narwhals-dev/narwhals).
Fork the repository by clicking on the fork button. You can find it in the right corner on the top of the page.

### 3. Clone the repository

Go to the forked repository on your GitHub account - you'll find it on your account in the tab Repositories. 
Click on the green `Code` button and then click the `Copy url to clipboard` icon.
Open a terminal, choose the directory where you would like to have Narwhals repository and run the following git command:

```bash
git clone <url you just copied>
```

for example:

```bash
git clone git@github.com:YOUR-GITHUB-USERNAME/narwhals.git narwhals-dev
```

You should then navigate to the folder you just created:

```bash
cd narwhals-dev
```

### 4. Add the `upstream` remote and fetch from it

```bash
git remote add upstream git@github.com:narwhals-dev/narwhals.git
git fetch upstream
```

Check to see the remote has been added with `git remote -v`, you should see something like this:

```bash
git remote -v                                                          
origin   git@github.com:YOUR-GITHUB-USERNAME/narwhals.git (fetch)
origin   git@github.com:YOUR-GITHUB-USERNAME/narwhals.git (push)
upstream git@github.com:narwhals-dev/narwhals.git (fetch)
upstream git@github.com:narwhals-dev/narwhals.git (push)
```

where `YOUR-GITHUB-USERNAME` will be your GitHub user name.

### 5. Setting up your environment

Here's how you can set up your local development environment to contribute.

#### Prerequisites for PySpark tests

If you want to run PySpark-related tests, you'll need to have Java installed. Refer to the [Spark documentation](https://spark.apache.org/docs/latest/#downloading) for more information.

#### Option 1: Use UV (recommended)

1. Make sure you have Python3.12 installed, create a virtual environment,
   and activate it. If you're new to this, here's one way that we recommend:
   1. Install uv (see [uv getting started](https://github.com/astral-sh/uv?tab=readme-ov-file#getting-started))
      or make sure it is up-to-date with:

      ```terminal
      uv self update
      ```

   2. Install Python3.12:

      ```terminal
      uv python install 3.12
      ```

   3. Create a virtual environment:

      ```terminal
      uv venv -p 3.12 --seed
      ```

   4. Activate it. On Linux, this is `. .venv/bin/activate`, on Windows `.\.venv\Scripts\activate`.

2. Install Narwhals: `uv pip install -e . --group local-dev`. This will include fast-ish core libraries and dev dependencies.
   If you also want to test other libraries like Dask , PySpark, and Modin, you can install them too with
   `uv pip install -e ".[dask, pyspark, modin]" --group local-dev`.

The pre-commit tool is installed as part of the local-dev dependency group. This will automatically format and lint your code before each commit, and it will block the commit if any issues are found.

Static typing is run separately from `pre-commit`, as it's quite slow. Assuming you followed all the instructions above, you can run it with `make typing`.

#### Option 2: use python3-venv

1. Make sure you have Python 3.8+ installed. If you don't, you can check [install Python](https://realpython.com/installing-python/)
   to learn how. Then, [create and activate](https://realpython.com/python-virtual-environments-a-primer/)
   a virtual environment.
2. Then, follow steps 2-4 from above but using `pip install` instead of `uv pip install`.

### 6. Working on your issue

Create a new git branch from the `main` branch in your local repository.
Note that your work cannot be merged if the test below fail.
If you add code that should be tested, please add tests.

### 7. Running tests

- To run tests, run `pytest`. To check coverage: `pytest --cov=narwhals`
- To run tests on the doctests, use `pytest narwhals --doctest-modules`
- To run unit tests and doctests at the same time, run `pytest tests narwhals --cov=narwhals --doctest-modules`
- To run tests multiprocessed, you may also want to use [pytest-xdist](https://github.com/pytest-dev/pytest-xdist) (optional)
- To choose which backends to run tests with you, you can use the `--constructors` flag:
  - To only run tests for pandas, Polars, and PyArrow, use `pytest --constructors=pandas,pyarrow,polars`
  - To run tests for all CPU constructors, use `pytest --all-cpu-constructors`
  - By default, tests run for pandas, pandas (PyArrow dtypes), PyArrow, and Polars.
  - To run tests using `cudf.pandas`, run `NARWHALS_DEFAULT_CONSTRUCTORS=pandas python -m cudf.pandas -m pytest`
  - To run tests using `polars[gpu]`, run `NARWHALS_POLARS_GPU=1 pytest --constructors=polars[lazy]`

### Backend-specific advice

- pandas:

  - Don't use `apply` or `map`. The only place we currently use `apply` is in `group_by` for operations
    which the pandas API just doesn't support, and even then, it's accompanied by a big warning.
  - Don't use inplace methods, unless you're creating a new object and are sure it's safe to modify
    it. In particular, you should never ever modify the user's input data.
  - Please remember that `assign`, `drop`, `reset_index`, and `rename`, though seemingly harmless, make
    full copies of the input data.

    - Instead of `assign`, prefer using `with_columns` at the compliant level.
    - For `drop` and `reset_index`, use `inplace=True`, so long as you're only modifying a new object which
      you created. Again, you should never modify user input. This may need updating if/when it's
      deprecated/removed, but please keep it for older pandas versions
      https://github.com/pandas-dev/pandas/pull/51466/files.
    - Instead of `rename`, prefer `alias` at the compliant level.

- Polars:

  - Never use `map_elements`.

- DuckDB / PySpark / anything lazy-only:

  - Never assume that your data is ordered in any pre-defined way.
  - Never materialise your data (only exception: `collect`).
  - Avoid calling the schema / column names unnecessarily.
  - For DuckDB, use the Python API as much as possible, only falling
    back to SQL as the last resort for operations not yet supported
    in their Python API (e.g. `over`).

### Test Failure Patterns

We aim to use three standard patterns for handling test failures:

Note: While we're not currently totally consistent with these patterns, any efforts towards our aim are appreciated and welcome.

1. `requests.applymarker(pytest.mark.xfail)`: Used for features that are planned but not yet supported. 
   ```python
   def test_future_feature(request):
       request.applymarker(pytest.mark.xfail)
       # Test implementation for planned feature
   ```

2. `pytest.mark.skipif`: Used when there's a condition under which the test cannot run (e.g., unsupported pandas versions).
   ```python
   @pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="requires pandas 2.0+")
   def test_version_dependent():
       # Test implementation
   ```

3. `pytest.raises`: Used for testing that code raises expected exceptions.
   ```python
   def test_invalid_input():
       with pytest.raises(ValueError, match="expected error message"):
           # Code that should raise the error
   ```

Document clear reasons in test comments for any skip/xfail patterns to help maintain test clarity.

If you want to have less surprises when opening a PR, you can take advantage of [nox](https://nox.thea.codes/en/stable/index.html) to run the entire CI/CD test suite locally in your operating system.

To do so, you will first need to install nox and then run the `nox` command in the root of the repository:

```bash
python -m pip install nox  # python -m pip install "nox[uv]"
nox
```

Notice that nox will also require to have all the python versions that are defined in the `noxfile.py` installed in your system.

#### Hypothesis tests

We use Hypothesis to generate some random tests, to check for robustness.
To keep local test suite times down, not all of these run by default - you can
run them by passing the `--runslow` flag to PyTest.

#### Testing Dask and Modin

To keep local development test times down, Dask and Modin are excluded from dev
dependencies, and their tests only run in CI. If you install them with

```terminal
uv pip install -U dask[dataframe] modin
```

then their tests will run too.

#### Testing cuDF

We can't currently test in CI against cuDF, but you can test it manually in Kaggle using GPUs. Please follow this [Kaggle notebook](https://www.kaggle.com/code/marcogorelli/testing-cudf-in-narwhals) to run the tests.

### Static typing

We run both `mypy` and `pyright` in CI. Both of these tools are included when installing Narwhals with the local-dev dependency group.

Run them with:
- `mypy narwhals tests`
- `pyright narwhals tests`

to verify type completeness / correctness.

Note that:
- In `_pandas_like`, we type all native objects as if they are pandas ones, though
  in reality this package is shared between pandas, Modin, and cuDF.
- In `_spark_like`, we type all native objects as if they are SQLFrame ones, though
  in reality this package is shared between SQLFrame and PySpark.

### 8. Writing the doc(strings)

If you are adding a new feature or changing an existing one, you should also update the documentation and the docstrings
to reflect the changes.

Writing the docstring in Narwhals is not an exact science, but we have some high level guidelines (if in doubt just ask us in the PR):

- The examples should be clear and to the point.
- The examples should import _one_ dataframe library, create a dataframe and exemplify the Narwhals functionality.
- We strive for balancing the use of different backend across all our docstrings examples.
- There are exceptions to the above rules!

Here an example of a docstring:

```python
>>> import pyarrow as pa
>>> import narwhals as nw
>>> df_native = pa.table({"foo": [1, 2], "bar": [6.0, 7.0]})
>>> df = nw.from_native(df_native)
>>> df.estimated_size()
32
```

Full discussion at [narwhals#1943](https://github.com/narwhals-dev/narwhals/issues/1943).

### 9. Building the docs

To build the docs, run `mkdocs serve`, and then open the link provided in a browser.
The docs should refresh when you make changes. If they don't, press `ctrl+C`, and then
do `mkdocs build` and then `mkdocs serve`.

### 10. Pull requests

When you have resolved your issue, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) in the Narwhals repository.

Please adhere to the following guidelines:

1. Start your pull request title with a [conventional commit](https://www.conventionalcommits.org/) tag. This helps us add your contribution to the right section of the changelog. We use "Type" from the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type).

   **TLDR**: The PR title should start with any of these abbreviations:
      `build`, `chore`, `ci`, `depr`, `docs`, `feat`, `fix`, `perf`, `refactor`, `release`, `test`.
      Add a `!`at the end, if it is a breaking change. For example `refactor!`.

2. This text will end up in the [changelog](https://github.com/narwhals-dev/narwhals/releases).
3. Please follow the instructions in the pull request form and submit.

## Working with Codespaces

Codespaces is a great way to work on Narwhals without the need of configuring your local development environment.
Every GitHub.com user has a monthly quota of free use of GitHub Codespaces, and you can start working in a codespace without providing any payment details.
You'll be informed per email if you'll be close to using 100% of included services.
To learn more about it visit [GitHub Docs](https://docs.github.com/en/codespaces/overview)

### 1. Make sure you have GitHub account

If you're new to GitHub, you'll need to create an account on [GitHub.com](https://github.com/) and verify your email address.

### 2. Fork the repository

Go to the [main project page](https://github.com/narwhals-dev/narwhals).
Fork the repository by clicking on the fork button. You can find it in the right corner on the top of the page.

### 3. Create codespace

Go to the forked repository on your GitHub account - you'll find it on your account in the tab Repositories.
Click on the green `Code` button and navigate to the `Codespaces` tab.
Click on the green button `Create codespace on main` - it will open a browser version of VSCode,
with the complete repository and git installed.
You can now proceed with the steps [5. Setting up your environment](#5-setting-up-your-environment) up to [10. Pull request](#10-pull-requests)
listed above in [Working with local development environment](#working-with-local-development-environment).

## How it works

If Narwhals looks like underwater unicorn magic to you, then please read
[how it works](https://narwhals-dev.github.io/narwhals/how_it_works/).

## Imports

In Narwhals, we are very particular about imports. When it comes to importing
heavy third-party libraries (pandas, NumPy, Polars, etc...) please follow these rules:

- Never import anything to do `isinstance` checks. Instead, just use the functions
  in `narwhals.dependencies` (such as `is_pandas_dataframe`);
- If you need to import anything, do it in a place where you know that the import
  is definitely available. For example, NumPy is a required dependency of PyArrow,
  so it's OK to import NumPy to implement a PyArrow function - however, NumPy
  should never be imported to implement a Polars function. The only exception is
  for when there's simply no way around it by definition - for example, `Series.to_numpy`
  always requires NumPy to be installed.
- Don't place a third-party import at the top of a file. Instead, place it in the
  function where it's used, so that we minimise the chances of it being imported
  unnecessarily.

We're trying to be really lightweight and minimal-overhead, and
unnecessary imports can slow things down.

## Claiming issues

If you start working on an issue, it's usually a good idea to let others know about this
in order to avoid duplicating work.

Do:

- When you're about to start working on an issue, and have understood the requirements
  and have some idea of what a solution would involve, comment "I've started working on this".
- Push partial work (even if unfinished) to a branch, which you can open in "draft" state.
- If someone else has commented that they're working on an issue but hadn't made any public
  work for 1-2 weeks, it's usually OK to assume that they're no longer working on it.

Don't:

- Don't claim issues that you intend to work on at a later date. For example, if it's Monday and
  you see an issue that interests you and you would like to work on it on Sunday, then the
  correct time to write "I'm working on this" is on Sunday when you start working on it.
- Don't ask for permission to work on issues, or to be assigned them. You have permission, we
  accept (and welcome!) contributions from everyone!

## Behaviour towards other contributors or maintainers

Above all else, please assume good intentions and go the extra mile to be super-extra-nice.

Some general guidelines:

- If in doubt, err on the side of being warm rather than being cold.
- If in doubt, put one extra positive emoji than one fewer one.
- Never delete or dismiss other maintainers' comments.
- Non-maintainers' comments should only be deleted if they are unambiguously spam
  (e.g. crypto adverts). In cases of rude or abusive behaviour, please contact the
  project author (`@MarcoGorelli`).
- Avoid escalating conflicts. People type harder than they speak, and online discourse
  is especially difficult. Again, please assume good intentions.

## Happy contributing!

Please remember to abide by the code of conduct, else you'll be conducted away from this project.

## Community Calendar

We have a community call every 2 weeks, all welcome to attend.

[Subscribe to our calendar](https://calendar.google.com/calendar/embed?src=27ff6dc5f598c1d94c1f6e627a1aaae680e2fac88f848bda1f2c7946ae74d5ab%40group.calendar.google.com).
