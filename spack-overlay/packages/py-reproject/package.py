from spack.package import *


class PyReproject(PythonPackage):
    """Astronomical image reprojection (astropy-affiliated)."""

    homepage = "https://github.com/astropy/reproject"
    git = "https://github.com/astropy/reproject.git"

    maintainers("karabo")
    license("BSD-3-Clause")

    # Pin to 0.9.1 to satisfy RASCIL constraints
    version("0.9.1", tag="v0.9.1")

    # Use default PythonPackage build backend selection in this Spack version

    # Python and build deps
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-cython@0.29:", type="build")
    depends_on("py-extension-helpers@1.0:", type="build")
    depends_on("py-numpy@1.18:", type=("build", "run"))
    depends_on("py-astropy@5:", type=("build", "run"))
    depends_on("py-scipy@1.5:", type=("build", "run"))
    depends_on("py-matplotlib@3.3:", type=("build", "run"))
    depends_on("py-pillow@8:", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    depends_on("py-astropy-healpix@1:", type=("build", "run"))

    import_modules = ["reproject"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import reproject; print(reproject.__version__) ")

    def test_healpix_api(self):
        python = which("python3") or which("python")
        if python:
            code = (
                "from reproject.healpix import reproject_from_healpix; print('ok')"
            )
            python("-c", code)


