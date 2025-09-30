from llnl.util.filesystem import filter_file

from spack.package import *


class PyAstropyHealpix(PythonPackage):
    """Python wrapper for HEALPix with Astropy integration.

    This package provides HEALPix functionality with an interface compatible
    with the Astropy ecosystem.
    """

    homepage = "https://github.com/astropy/astropy-healpix"
    pypi = "astropy-healpix/astropy-healpix-1.0.0.tar.gz"
    git = "https://github.com/astropy/astropy-healpix.git"

    license("BSD-3-Clause")

    # Use git tag to avoid requiring an sdist checksum; docker build passes --no-checksum
    version("1.0.0", tag="v1.0.0")
    version("1.1.2", tag="v1.1.2")

    # Ensure we use the pip/PEP-517 build backend consistently
    build_system("python_pip")

    # Python and runtime dependencies (aligned broadly with Astropy ecosystem)
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    # Required by astropy-healpix PEP517 build metadata (version inference)
    depends_on("py-setuptools-scm@6:", type="build")
    # Required by astropy-style builds to locate/compile extensions cleanly
    depends_on("py-extension-helpers@1:", type="build")
    # Make sure wheel/pip are present for PEP517 builds in this Spack release
    depends_on("py-wheel", type="build")
    depends_on("py-pip@21:", type="build")

    depends_on("py-numpy@1.18:", type=("build", "run"))
    depends_on("py-astropy@5:", type=("build", "run"))
    depends_on("py-pyyaml", type=("build", "run", "test"))

    import_modules = ["astropy_healpix"]

    def setup_build_environment(self, env):
        # Avoid setuptools_scm attempting to fetch VCS by providing the version
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPY_HEALPIX", self.spec.version.string)
        # Ensure we do not create isolated venvs that miss Spack-provided deps
        env.set("PIP_NO_BUILD_ISOLATION", "1")

    def patch(self):
        if self.spec.satisfies("@1.1.2"):
            filter_file("numpy>=2.0.0", "numpy>=1.20", "pyproject.toml")

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


