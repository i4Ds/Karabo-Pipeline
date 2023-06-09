import glob

import pytest


@pytest.mark.test
def test_examples():
    """Test all examples in the documentation"""
    # get all example scripts
    example_scripts = glob.glob("doc/src/examples/_example_scripts/*.py")
    # run all example scripts
    for example_script in example_scripts:
        try:
            exec(open(example_script).read())
        except Exception as e:
            pytest.fail(
                "Example script {} failed with error: {}".format(example_script, e)
            )
