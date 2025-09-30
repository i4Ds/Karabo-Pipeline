from spack.package import *


class PySeqfile(PythonPackage):
    """Small utilities for working with sequence files.

    Provides lightweight helpers for reading/writing simple sequence files.
    """

    homepage = "https://pypi.org/project/seqfile/"
    pypi = "seqfile/seqfile-0.2.0.tar.gz"

    version("0.2.0", sha256="3e688d2777f6a8c8d1515b93a6c0f10b68a02653d52497a21f4345c3e74cda48")

    # Spack 0.23-compatible backends
    build_system("python_pip", "python_setuptools")

    # Python & build deps
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools", type=("build", "run", "test"))
    depends_on("py-wheel", type="build")
    depends_on("py-natsort", type=("build", "run"))

    import_modules = ["seqfile"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import seqfile; print('py-seqfile OK') ")


