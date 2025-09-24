from spack.package import *


class PyAstropyHealpix(PythonPackage):
    """Python wrapper for HEALPix with Astropy integration.

    This package provides HEALPix functionality with an interface compatible
    with the Astropy ecosystem.
    """

    homepage = "https://github.com/astropy/astropy-healpix"
    pypi = "astropy-healpix/astropy-healpix-1.0.0.tar.gz"
    git = "https://github.com/astropy/astropy-healpix.git"

    maintainers("karabo")
    license("BSD-3-Clause")

    # Use git tag to avoid requiring an sdist checksum; docker build passes --no-checksum
    version("1.0.0", tag="v1.0.0")

    # Use default PythonPackage build backend selection in this Spack version

    # Python and runtime dependencies (aligned broadly with Astropy ecosystem)
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")

    depends_on("py-numpy@1.18:", type=("build", "run"))
    depends_on("py-astropy@5:", type=("build", "run"))
    depends_on("py-pyyaml", type=("build", "run", "test"))

    import_modules = ["astropy_healpix"]

    def test_import(self):
        """Verify the module imports after installation."""
        python = which("python3") or which("python")
        if python:
            python("-c", "import astropy_healpix as ah; print(ah.__version__) ")

    def test_basic_api(self):
        """Lightweight runtime check of a basic API call."""
        python = which("python3") or which("python")
        if python:
            code = (
                "from astropy_healpix import HEALPix; from astropy import units as u; "
                "hp=HEALPix(nside=1, order='nested', frame=None); print(hp.nside)"
            )
            python("-c", code)


