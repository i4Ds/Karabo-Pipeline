from spack.package import *


class PyAstroplan(PythonPackage):
    """Observation planning package for astronomy.

    Provides utilities to plan observation schedules, compute target
    visibility, and coordinate transforms, built on top of Astropy.
    """

    homepage = "https://astroplan.readthedocs.io/"
    git = "https://github.com/astropy/astroplan.git"

    version("0.8", tag="v0.8")
    version("0.10.1", tag="v0.10.1")

    # Allow pip/setuptools backends (Spack 0.23 compatible)
    build_system("python_pip", "python_setuptools")

    # Build dependencies
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    depends_on("py-packaging", type="build")

    # Runtime dependencies
    depends_on("py-astropy@4:", type=("build", "run"))
    depends_on("py-numpy@1.20:", type=("build", "run"))
    depends_on("py-scipy@1.5:", type=("build", "run"))
    depends_on("py-pyyaml", type=("build", "run", "test"))
    depends_on("py-six", type=("build", "run", "test"))
    depends_on("py-pytz", type=("build", "run", "test"))

    import_modules = ["astroplan"]

    def setup_build_environment(self, env):
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPLAN", self.spec.version.string)

    def test_import(self):
        python = self.spec["python"].command
        if python:
            python("-c", "import astroplan as ap; print('ok') ")


