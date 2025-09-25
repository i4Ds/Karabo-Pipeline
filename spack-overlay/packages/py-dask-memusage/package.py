from spack.package import *


class PyDaskMemusage(PythonPackage):
    """Low-impact, task-level memory profiling for Dask."""

    homepage = "https://pypi.org/project/dask-memusage/"
    # PyPI provides a wheel (no sdist). Fetch the wheel directly.

    maintainers("karabo")

    version("1.1")

    def url_for_version(self, version):
        ver = str(version)
        # Wheel filenames use underscore in project name
        return f"https://files.pythonhosted.org/packages/py3/d/dask_memusage/dask_memusage-{ver}-py3-none-any.whl"

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

    def install(self, spec, prefix):
        # Install the fetched wheel using pip
        python = spec["python"].command
        archive = getattr(self.stage, "archive_file", None)
        if archive:
            python("-m", "pip", "install", "--no-build-isolation", "--no-deps", f"--prefix={prefix}", archive)
        else:
            python("-m", "pip", "install", "--no-build-isolation", "--no-deps", f"--prefix={prefix}", f"dask_memusage=={self.version}")


