from spack.package import *


class PyDucc(PythonPackage):
    """Python bindings for DUCC (ducc0)."""

    homepage = "https://gitlab.mpcdf.mpg.de/mtr/ducc/"
    pypi = "ducc0/ducc0-0.27.0.tar.gz"

    maintainers("karabo")
    license("BSD-2-Clause")

    version("0.27.0")

    # Use default PythonPackage backend selection for this Spack version

    # Build deps
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-pybind11", type="build")
    depends_on("py-packaging", type="build")
    depends_on("py-numpy@1.18:", type=("build", "run"))

    import_modules = ["ducc0"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import ducc0, sys; print(ducc0.__version__) ")


