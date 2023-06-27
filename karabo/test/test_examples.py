import glob
import os

import pytest


def test_examples():
    """Test all examples in the documentation"""
    # get all example scripts
    glob_str = os.path.join("doc", "src", "examples", "_example_scripts", "*.py")
    example_scripts = glob.glob(glob_str)
    # run all example scripts
    for example_script in example_scripts:
        try:
            exec(open(example_script).read())
        except Exception as e:
            pytest.fail(
                "Example script {} failed with error: {}".format(example_script, e)
            )
