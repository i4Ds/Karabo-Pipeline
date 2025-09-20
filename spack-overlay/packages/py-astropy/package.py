from spack.package import *


class PyAstropy(PythonPackage):
    """The core package for Astronomy in Python."""

    homepage = "https://www.astropy.org/"
    pypi = "astropy/astropy-5.3.tar.gz"
    git = "https://github.com/astropy/astropy.git"

    maintainers("karabo")

    version("5.3.4", sha256="d490f7e2faac2ccc01c9244202d629154259af8a979104ced89dc4ace4e6f1d8")
    version("5.2.2", sha256="e6a9e34716bda5945788353c63f0644721ee7e5447d16b1cdcb58c48a96b0d9c")
    version("5.1.1", sha256="ba4bd696af7090fd399b464c704bf27b5633121e461785edc70432606a94bd81")

    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-setuptools-scm@6.2:", type="build")
    depends_on("py-cython@0.29:3.0", type="build")
    depends_on("py-extension-helpers@1.0:", type=("build", "run"))
    depends_on("py-packaging@21:", type=("build", "run"))
    depends_on("py-numpy@1.18:", type=("build", "run"))
    depends_on("py-numpy@1.22:", type=("build", "run"), when="@5.3:")
    depends_on("py-pyerfa@2.0:", type=("build", "run"), when="@5.1:")
    depends_on("pkgconfig", type="build")

    # Disable default import tests to avoid heavy test imports during install
    import_modules = []