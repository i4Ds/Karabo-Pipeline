from spack.package import *


class PyPhotutils(PythonPackage):
    """Photometry tools for Python.

    Astropy-affiliated package for image photometry.
    """

    homepage = "https://github.com/astropy/photutils"
    git = "https://github.com/astropy/photutils.git"

    maintainers("karabo")
    license("BSD-3-Clause")

    version("1.11.0", tag="1.11.0")

    # Default backend selection for this Spack

    # Build deps
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-build", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-cython@0.29:", type="build")
    depends_on("py-extension-helpers@1.0:", type="build")

    # Run deps
    depends_on("py-astropy@5.1:", type=("build", "run"))
    depends_on("py-numpy@1.22:", type=("build", "run"))
    depends_on("py-pyyaml", type=("build", "run", "test"))

    import_modules = ["photutils"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import photutils, sys; print(photutils.__version__) ")
