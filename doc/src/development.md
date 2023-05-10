# Development Documentation

## Setup local dev environment

First clone the code via git.
Then create a local development environment with the provided `environment.yaml` file.

```shell
conda env create -n <your-env-name> -f environment.yaml
```

Then install the development dependencies using `requirements.txt`.

```shell
conda activate <your-env-name>
pip install -r requirements.txt
```

With this only the dependencies but not the current version of karabo will be installed into a conda environment.
Then you can simply run your code inside that environment. To tell Python to treat the reposity as a package, the following links can be helpful:

[Setup Python Interpreter in PyCharm](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html).
[how to use conda develop?](https://github.com/conda/conda-build/issues/1992)

## Formatting

To increase the readability of the code and to better detect potential errors already during development, a number of tools are used. These tools must first be installed in the virtual environment using `pip install -r requirements.txt`. If possible use the versions defined in `requirements.txt`, so that all developers work with the same versions. The configurations of the tools are handled in `setup.cfg`. If changes to the configurations are desired, the team members should agree to this (e.g. via a meeting).

It is possible that certain tools complain about something that is not easy or even impossible to fix. ONLY then, there are options to ignore certain lines of code or even whole files for the checker. E.g. `# noqa` ignores inline flake8 complaints. But be careful not to accidentally ignore the whole file (e.g. with `# flake8: noqa`). Please refer to the documentation of the respective tool to learn how to ignore the errors.

We recommend the setup using [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks), so that the in `.pre-commit-config.yaml` specified tools prevent commits if the checks on the added files fail. The hook can be easily initialized using `pre-commit install` in the root directory of the repository. If absolutely necessary, the hooks can be ignored using the `--no-verify` flag in the git-commands. However, some checks are included in the CI-tests, so you have to check your files anyway. The [pre-commit-configs](https://pre-commit.com/) used for this are defined in `.pre-commit-config.yaml`.

The following sections list the tools to be used and their function & usage.

### black

[https://github.com/psf/black](black) is a Python [PEP 8](https://peps.python.org/pep-0008/) compliant opinionated code formatter. It formats entire Python files with the CLI command `black {source_file_or_directory}`. For specific options, see [command line options](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html) or `black --help`.

### isort

[isort](https://pycqa.github.io/isort/) organizes Python imports alphabetically and automatically separated into sections and by type. The command can be called via `isort {source_file}`. There are further possibilities in the [documentation](https://pycqa.github.io/isort/) to apply isort to multiple files at once.

### mypy

[mypy](https://mypy.readthedocs.io/en/stable/) is a static type checker for Python. It ensures that variables and functions are used correctly in the code. It warns the developer if type hints ([PEP 484](https://peps.python.org/pep-0484/)) are not used correctly. The CLI `mypy {source_file_or_directory}` makes it easy to use mypy (although it may take a while). For specific options, see [command line options](https://mypy.readthedocs.io/en/stable/command_line.html) or `mypy --help`. The configs set in `setup.cfg` are `--strict`, so there is no need to set the flag by yourself.

If you run mypy several times for the same file(s) and it takes a while, you can launch the mypy daemon `dmypy start` and then check your file(s) using `dmypy check {file_or_directory}`. The first time you check might still take a while. However, once the check is done and you want to check a file again, the check will only take into account the changes you have made, so it will be much faster.

### pydocstyle

[pydocstyle](http://www.pydocstyle.org/en/stable/) supports docstring checking out of the box. The check can be used via CLI `pydocstyle {source_file_or_directory}. Further options can be found in the [documentation](http://www.pydocstyle.org/en/stable/usage.html) or via `pydocstyle --help`.

In our project we use the [google convention](http://www.pydocstyle.org/en/stable/error_codes.html?highlight=google#default-conventions). Python files, classes as well as functions need corresponding docstrings. There are autocompletion tools for IDE's that create docstrings in the correct format (e.g. VSCode [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)). Below is an example of a dummy function with the corresponding docstring template:

```
def fun(arg1: int, arg2: int) -> int:
    """__summary__

    __detailed_description_if_needed__

    Args:
        some_arg: __description__

    Returns:
        __description__
    """
    arg_added = arg1 + arg2
    return arg_added
```

### flake8

[flake8](https://flake8.pycqa.org/en/latest/) is a tool for style guide enforcement. It checks various style errors such as unused import, variables, maximum line length, spacing and others. The style-checker can be used via the CLI `flake8 {source_file_or_directory}`. In addition, you can set your IDE's **Linter to flake8**, so that the errors are displayed directly (it might be necessary to install an extension to do so).

It is recommended to run *black* before manually checking with *flake8* because *black* might fix a lot of *flake8* related issues. It is possible that the settings of *isort* and *black* are not compatible with *flake8* and therefore should get changed.

## Upload Objects to CSCS

Make objects available through CSCS object-storage REST-API. The GET-request URI follows the following format: `{CSCS-object-store-prefix}/{container}/{object}` where the prefix comes from [CSCS API access](https://castor.cscs.ch/dashboard/project/api_access/).

### Through Web Interface

1. Go to [Castor](https://castor.cscs.ch/) and authenticate yourself.
2. Navigate to `project/object-storage/container` and choose the container you want to upload (e.g. `karabo_public`). 
3. Click the upload option and upload the file of choice.

### Through CLI

1. Read https://github.com/eth-cscs/openstack/tree/master/cli
2. `source openstack/cli/castor.env`
3. `swift post karabo_public --read-acl ".r:*,.rlistings"`
4. `swift upload karabo_public example_file.fits`

## Update documentation

The docs are built from the python source code and other doc source files located in /doc/src.
The .rst and .md files need to be referenced somehow inside of index.rst or an already referenced page inside of index.rst to be viewable by the public upon building the documentation

If you want to add any sort of extra text like examples or other sort of documentation do this by adding a new .md file or .rst file inside of the /doc/src folder.
Then write your text and then add the new file to index.rst in the topmost toctree. 

````rst


Welcome to Karabo-Pipeline's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Container
   modules
   examples/Examples.md
   --> ADD THE REFERENCE TO YOUR FILE HERE


````

Also subfiles can also point to other files like the file type defines. 
So an md file can reference like ``[some file](path/to/some/file)``.

When adding new submodules or modules. You need to update the modules.rst file accordingly and add new files similiar to the karabo.simulation.rst. To enable the automatic generation of the documentation via the python docstrings.
There is also the command ```sphinx-apidoc``` from sphinx (our doc engine), that can automate this.

If you want to work this sphinx locally on your machine, for example to use this sphinx-apidoc command. Thus, use the following commands to generate the documentation:

```shell
pip install -r requirements.txt
make html
```

## Update Tests

We use the basic ``unittest`` python package ([unittest docs](https://docs.python.org/3/library/unittest.html)).
The unit tests are run automatically on every push.
To add a new test simply go to the karabo/test folder.
Add a new file for a new set of tests in this shape.

```python
class TestSimulation(unittest.TestCase):

    def someTest(self):
        # run some code that tests some functionality
        result = someFunction()
        # use the assertion functions on self
        self.assertEquals(result, 4)
```

Add tests for when you write some sort of new code that you feel like might break.


TIP:
If you validate your code manually, consider just writing a method in a test class instead of opening a jupyter notebook and writing a new cell or a terminal window where you would execute the code you want to test.

## Create a Release
When everything is merged which should be merged, a new Release can be deployed on `conda-forge` as following:
- [Karabo-Pipline | Releases](https://github.com/i4Ds/Karabo-Pipeline/releases)
- Click on `Draft a new release`
- Define a Version by clicking `Choose a tag`. Currently we increment the second number by 1.
- Update version in `karabo/version.py`
- Check that the `Target` is set to `main`.
- Describe the release (get inspired by the previous releases).
- Click `Publish release`. 
- Check on [Karabo-Pipeline | Github Actions](https://github.com/i4Ds/Karabo-Pipeline/actions) that the release is succesful. 
- Check that the new version is on [Anaconda.org | Packages](https://anaconda.org/i4ds/karabo-pipeline)

