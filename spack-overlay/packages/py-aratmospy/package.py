"""Spack recipe for ARatmospy.

This file uses Spack's DSL (version, depends_on, etc.), which confuses static
linters. Disable lints for this file.
"""  # flake8: noqa  # mypy: ignore-errors
# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportMissingModuleSource=false

from spack.package import (version, build_system, PythonPackage, depends_on)

class PyAratmospy(PythonPackage):
    """Autoregressive atmosphere generator (legacy setuptools packaging).

    This overlay builds a specific git commit to match reproducible environments.
    """

    homepage = "https://github.com/i4Ds/ARatmospy"
    git = "https://github.com/i4Ds/ARatmospy.git"

    # Pin the exact commit used in docker builds
    version("1.0.0", commit="67c302a136beb40a1cc88b054d7b62ccd927d64f", preferred=True)

    # Use pip-based build system (compatible with legacy setup.py projects)
    build_system("python_pip")

    # Importable top-level module is capitalized in this project
    import_modules = ["ARatmospy"]

    # Runtime requirements from setup.py (keep relaxed to allow environment pins)
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-pip", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-scipy@1.9:", type=("build", "run"))
    depends_on("py-astropy", type=("build", "run"))
    depends_on("py-matplotlib", type=("build", "run"))
    depends_on("py-pyfftw", type=("build", "run"))

    def setup_build_environment(self, env):
        # Ensure Spack provides dependencies without pip attempting isolation
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Provide deterministic version metadata if upstream ever enables SCM tooling
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ARATMOSPY", self.spec.version.string)

    def test_small_screen_and_fits(self):
        """Run a tiny ARatmospy screen evolution and write a small FITS.

        This is a scaled-down version of ARatmospy's example test to keep
        runtime and memory usage minimal in Spack's test environment.
        """
        python = self.spec["python"].command
        code = (
            "import numpy as np\n"
            "from astropy.io import fits\n"
            "from astropy.wcs import WCS\n"
            "from ARatmospy.ArScreens import ArScreens\n"
            "screen_width_metres = 2e3\n"
            "r0 = 500.0\n"
            "bmax = 2e3\n"
            "sampling = 250.0\n"
            "m = int(bmax / sampling)  # 8\n"
            "n = int(screen_width_metres / bmax)  # 1\n"
            "num_pix = n * m  # 8\n"
            "pscale = screen_width_metres / (n * m)\n"
            "rate = 1.0\n"
            "alpha_mag = 0.99\n"
            "layer_params = np.array([(r0, 10.0, 0.0, 300e3)])\n"
            "num_times = 2\n"
            "sc = ArScreens(n, m, pscale, rate, layer_params, alpha_mag)\n"
            "sc.run(num_times)\n"
            "w = WCS(naxis=4); w.naxis = 4\n"
            "w.wcs.cdelt = [pscale, pscale, 1.0 / rate, 1.0]\n"
            "w.wcs.crpix = [num_pix // 2 + 1, num_pix // 2 + 1, num_times // 2 + 1, 1.0]\n"
            "w.wcs.ctype = ['XX', 'YY', 'TIME', 'FREQ']\n"
            "w.wcs.crval = [0.0, 0.0, 0.0, 1e8]\n"
            "data = np.zeros([1, num_times, num_pix, num_pix], dtype=np.float32)\n"
            "for i, screen in enumerate(sc.screens[0]):\n"
            "    data[:, i, ...] += screen[np.newaxis, ...].astype(np.float32)\n"
            "fits.writeto('test_screen_small.fits', data=data, header=w.to_header(), overwrite=True)\n"
            "print('ARATMOSPY_TEST_OK', data.shape)\n"
        )
        python("-c", code)
