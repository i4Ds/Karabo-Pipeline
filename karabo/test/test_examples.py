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
                subprocess.run([sys.executable, example_script], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "command '{}' return with error (code {}): {}".format(
                        e.cmd, e.returncode, e.output
                    )
                )
