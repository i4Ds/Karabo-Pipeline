# Development Documentation

# Setup local dev environment

First clone the code via git.
Then create a local development environment with the provided `environment-dev.yaml` file.

```shell
conda env create -f environment-dev.yaml
```

With this only the dependencies but not the current version of karabo will be installed into a conda environment.
Then you can simply run your code inside that environment. 
Or a bit more sophisticated. You can use PyCharm or any other Python IDE. Then inside of the IDE setup the correct python interpreter to be the one you just steup before with the ```conda env create``` command.

[Setup Python Interpreter in PyCharm](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html).


# Update documentation

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

If you want to work this sphinx locally on your machine, for example to use this sphinx-apidoc command. You can create a different conda environment with the ``environment.yaml`` file you find inside if /doc 

# Update Tests

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
