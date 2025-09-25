from spack.package import *


class PyDaskMemusage(PythonPackage):
    """Low-impact, task-level memory profiling for Dask."""

    homepage = "https://pypi.org/project/dask-memusage/"
    pypi = "dask-memusage/dask-memusage-1.1.tar.gz"

    maintainers("karabo")

    version("1.1", sha256="29d9f25074fecd7ca249e972cb3ec0b909a1dcefaf037c8d5fca24fadbf66757")

    build_system("python_pip", "python_setuptools")

    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools", type=("build", "run", "test"))
    depends_on("py-dask", type=("build", "run"))
    depends_on("py-psutil", type=("build", "run"))

    import_modules = ["dask_memusage"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import dask_memusage; print('py-dask-memusage OK') ")


