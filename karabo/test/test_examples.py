import glob
import unittest


class TestDocExamples(unittest.TestCase):
    def test_examples(self):
        """Test all examples in the documentation"""
        # get all example scripts
        example_scripts = glob.glob("doc/src/examples/_example_scripts/*.py")
        # run all example scripts
        for example_script in example_scripts:
            try:
                exec(open(example_script).read())
            except Exception as e:
                self.fail(
                    "Example script {} failed with error: {}".format(example_script, e)
                )
