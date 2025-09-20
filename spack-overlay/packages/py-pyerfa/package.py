from spack.package import *

class PyPyerfa(PythonPackage):
    """
    PyERFA is the Python wrapper for the ERFA library (Essential
    Routines for Fundamental Astronomy), a C library containing key
    algorithms for astronomy, which is based on the SOFA library
    published by the International Astronomical Union (IAU). All C
    routines are wrapped as Numpy universal functions, so that they
    can be called with scalar or array inputs.
    """

    homepage = "https://github.com/liberfa/pyerfa"
    pypi = "pyerfa/pyerfa-2.0.0.1.tar.gz"

    maintainers("karabo")

    version("2.0.0.1", sha256="2fd4637ffe2c1e6ede7482c13f583ba7c73119d78bef90175448ce506a0ede30")

    depends_on("python@3.7:", type=("build", "run"))
    # Pin to numpy 1.23.5 to match our environment and avoid ABI issues
    depends_on("py-numpy@1.23.5", type=("build", "run"))
    depends_on("py-setuptools@42:", type="build")
    depends_on("py-packaging", type="build")
    depends_on("py-jinja2@2.10.3:", type="build")
    depends_on("erfa", type=("build", "link", "run"))
    depends_on("pkgconfig", type="build")

    # Disable Spack's default import_module tests; we run our own below
    import_modules = []

    def setup_build_environment(self, env):
        """Set up build environment to find erfa."""
        spec = self.spec
        env.set("LDFLAGS", spec['erfa'].libs.ld_flags)
        env.set("CFLAGS", spec['erfa'].headers.include_flags)

    def test(self):
        """Self-contained test; works without a Spack view."""
        import os
        import subprocess

        python = self.spec["python"].command.path

        # Compute site-packages for the just-installed package and prepend to PYTHONPATH
        py_ver = str(self.spec["python"].version.up_to(2))
        site_dir = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{site_dir}:{env.get('PYTHONPATH','')}" if site_dir else env.get("PYTHONPATH", "")

        test_code = """
import sys
# Always ensure we can import the Python module installed by this package
try:
    import erfa
except Exception as exc:
    print(f"erfa missing or not importable: {exc}", file=sys.stderr)
    sys.exit(1)

# If astropy is available, run the EarthLocation check; otherwise run a basic ERFA ufunc
try:
    from astropy import units as u
    from astropy.coordinates import EarthLocation, Longitude, Latitude
    lon = Longitude(116.76444824, unit=u.deg)
    lat = Latitude(-26.82472208, unit=u.deg)
    height = 300 * u.m
    loc = EarthLocation.from_geodetic(lon=lon, lat=lat, height=height)
    assert loc is not None
except Exception as astropy_exc:
    # Fallback: exercise a core ERFA ufunc with scalar inputs
    try:
        import numpy as np
        import erfa
        _xyz = erfa.gd2gc(1, 0.0, 0.0, 0.0)
    except Exception as erfa_exc:
        print(f"basic erfa check failed: {erfa_exc}", file=sys.stderr)
        sys.exit(1)
print("ok")
"""

        subprocess.run([python, "-c", test_code], check=True, env=env)
