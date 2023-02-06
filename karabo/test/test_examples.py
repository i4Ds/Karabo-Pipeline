import unittest
import os, sys, subprocess, glob


class TestDocExamples(unittest.TestCase):
    def test_examples(self):
        """Test all examples in the documentation"""
        # get all example scripts
        path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(path + "/../..")
        example_scripts = glob.glob("doc/src/examples/_example_scripts/*.py")
        # run all example scripts
        for example_script in example_scripts:
            try:
                output = subprocess.check_output(
                    [sys.executable, example_script], stderr=subprocess.STDOUT
                )
            except subprocess.CalledProcessError as e:
                output = e.output
                if b"Traceback" in output:
                    error = output[output.index(b"Traceback") :].decode()
                else:
                    error = "Unknown error"
                raise RuntimeError(
                    f'Example script "{example_script}" failed with error: {error}'
                )
