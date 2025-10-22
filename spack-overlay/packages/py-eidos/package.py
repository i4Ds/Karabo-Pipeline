"""Spack recipe for eidos.

This file uses Spack's DSL (version, depends_on, etc.), which confuses static
linters. Disable lints for this file.
"""  # flake8: noqa  # mypy: ignore-errors
# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportMissingModuleSource=false

from spack.package import (version, build_system, PythonPackage, depends_on)

class PyEidos(PythonPackage):
    """Modelling primary beams of radio telescope antennae.

    Primary beam modelling for radio astronomy antennas, with support for
    MeerKAT L-band beams from holographic and EM simulations.
    """

    homepage = "https://github.com/ratt-ru/eidos"
    git = "https://github.com/i4Ds/eidos.git"

    # Pin the exact commit used in docker builds
    version("1.1.0", commit="74ffe0552079486aef9b413efdf91756096e93e7", preferred=True)

    # Use pip-based build system (compatible with legacy setup.py projects)
    build_system("python_pip")

    # Importable top-level module
    import_modules = ["eidos"]

    # Runtime requirements from setup.py (keep relaxed to allow environment pins)
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-pip", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    depends_on("py-astropy", type=("build", "run"))
    depends_on("py-future", type=("build", "run"))

    def setup_build_environment(self, env):
        # Ensure Spack provides dependencies without pip attempting isolation
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Provide deterministic version metadata if upstream ever enables SCM tooling
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EIDOS", self.spec.version.string)

    def test_eidos_import(self):
        """Verify that eidos can be imported successfully."""
        python = self.spec["python"].command
        code = (
            "import eidos\n"
            "import eidos.create_beam\n"
            "import eidos.spatial\n"
            "import eidos.spectral\n"
            "import eidos.util\n"
            "print('EIDOS_IMPORT_OK')\n"
        )
        python("-c", code)

