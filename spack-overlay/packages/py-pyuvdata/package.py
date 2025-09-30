# from analysis of https://github.com/RadioAstronomySoftwareGroup/pyuvdata
from spack.package import PythonPackage


class PyPyuvdata(PythonPackage):
    """An interface for astronomical interferometric datasets in Python."""

    homepage = "https://github.com/RadioAstronomySoftwareGroup/pyuvdata"
    pypi = "pyuvdata/pyuvdata-2.4.2.tar.gz"

    version("2.4.2", sha256="cec2a6630f3d3b39511548f94a557ec302057edd4096d5f99434e3d694291f2f")

    # Python support per upstream classifiers
    depends_on("python@3.9:3.12", type=("build", "run"))

    # Build system
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-wheel", type="build")
    # setup.py imports setuptools_scm and it is in install_requires as well
    depends_on("py-setuptools-scm@:6,7.0.3:", type=("build", "run"))

    # Cythonized extensions need Cython and NumPy headers at build time
    depends_on("py-cython@0.23:", type="build")
    depends_on("py-packaging", type="build")
    depends_on("py-numpy@1.20:", type=("build", "run"))

    # Core runtime requirements from install_requires
    depends_on("py-astropy@5.0.4:", type=("build", "run"))
    depends_on("py-docstring-parser@0.15:", type=("build", "run"))
    depends_on("py-h5py@3.1:", type=("build", "run"))
    depends_on("py-pyerfa@2.0:", type=("build", "run"))
    depends_on("py-scipy@1.5:", type=("build", "run"))

    # Optional extras mapped from pyuvdata_test_variants.sh for Spack variants
    variant("astroquery", default=False, description="Enable astroquery extra")
    variant("cst", default=True, description="Enable CST extra (pyyaml)")
    variant("hdf5_compression", default=False, description="Enable HDF5 compression extra")
    variant("healpix", default=True, description="Enable HEALPix extra")
    variant("lunar", default=False, description="Enable lunar extra")
    variant("novas", default=False, description="Enable NOVAS extra")
    variant("casa", default=True, description="Enable CASA (casacore) extra")

    # with when("+astroquery"):
    #     depends_on("py-astroquery@0.4.4:", type=("build", "run"))

    with when("+cst"):
        depends_on("py-pyyaml@5.3:", type=("build", "run"))

    # with when("+hdf5_compression"):
    #     depends_on("py-hdf5plugin@3.1.0:", type=("build", "run"))

    with when("+healpix"):
        depends_on("py-astropy-healpix@1.0:", type=("build", "run"))

    # with when("+lunar"):
    #     depends_on("py-lunarsky@0.2.1:", type=("build", "run"))

    # with when("+novas"):
    #     depends_on("py-novas@3.1:", type=("build", "run"))
    #     depends_on("py-novas-de405@3.1:", type=("build", "run"))

    with when("+casa"):
        depends_on("py-casacore@3.5.0:", type=("build", "run"))

    # Basic import test
    import_modules = ["pyuvdata"]

    def setup_build_environment(self, env):
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYUVDATA", self.spec.version.string)

    def test_astropy_erfa(self):
        """Regression test for astropy->erfa dtype handling.

        On the pyuvdata baseline image the downstream karabo tests surfaced a
        ValueError from erfa.gd2gc fed with Angle/Quantity inputs. Exercise the
        same EarthLocation path so the failure reproduces under `spack test run`.
        """
        python = self.spec["python"].command
        code = (
            "import sys\n"
            "from astropy import units as u\n"
            "from astropy.coordinates import EarthLocation\n"
            "try:\n"
            "    loc = EarthLocation.from_geodetic(\n"
            "        lon=116.76444824 * u.deg,\n"
            "        lat=-26.82472208 * u.deg,\n"
            "        height=300 * u.m,\n"
            "    )\n"
            "    print('EARTHLOCATION_OK', loc.x.value, loc.y.value, loc.z.value)\n"
            "except Exception as exc:\n"
            "    print('EARTHLOCATION_FAIL', repr(exc))\n"
            "    sys.exit(1)\n"
        )
        python("-c", code)