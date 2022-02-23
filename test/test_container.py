import unittest


class TestContainer(unittest.TestCase):

    def test_build(self):
        from karabo.Container import environment, container
        env = environment.Environment("test", 3.7, "python -V")
        env.addChannel("conda-forge")
        env.addChannel("i4ds")
        env.addPackage("oskar")
        env.addPackage("bdsf")
        container = container.Container(env)
        container.run()
