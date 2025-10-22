from spack.package import *


class PyDaskMemusage(PythonPackage):
    """Low-impact, task-level memory profiling for Dask."""

    homepage = "https://pypi.org/project/dask-memusage/"
    git = "https://github.com/itamarst/dask-memusage.git"
    pypi = "dask_memusage/dask_memusage-1.1.tar.gz"

    version("1.1", sha256="29d9f25074fecd7ca249e972cb3ec0b909a1dcefaf037c8d5fca24fadbf66757")

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
        python = self.spec["python"].command
        if python:
            python("-c", "import dask_memusage; print('py-dask-memusage OK') ")

    def install(self, spec, prefix):
        # Install the fetched wheel using pip
        python = spec["python"].command
        python_args = [
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            f"--prefix={prefix}",
        ]
        archive = getattr(self.stage, "archive_file", None)
        if archive:
            python_args.append(archive)
        else:
            python_args.append(f"dask_memusage=={self.version}")
        python(*python_args)


