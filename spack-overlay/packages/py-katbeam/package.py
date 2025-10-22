"""Spack recipe for katbeam.

This file uses Spack's DSL (version, depends_on, etc.), which confuses static
linters. Disable lints for this file.
"""  # flake8: noqa  # mypy: ignore-errors
# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportMissingModuleSource=false

from spack.package import (version, build_system, PythonPackage, depends_on)

class PyKatbeam(PythonPackage):
    """Karoo Array Telescope primary beam model library.

    Primary beam model library for the MeerKAT project, providing functionality to
    compute simplified beam patterns of MeerKAT antennas using the JimBeam class.
    """

    homepage = "https://github.com/ska-sa/katbeam"
    git = "https://github.com/ska-sa/katbeam.git"

    # Pin the exact commit used in docker builds
    version("0.1.0", commit="5ce6fcc35471168f4c4b84605cf601d57ced8d9e", preferred=True)

    # Use pip-based build system (compatible with legacy setup.py projects)
    build_system("python_pip")

    # Importable top-level module
    import_modules = ["katbeam"]

    # Runtime requirements from setup.py (keep relaxed to allow environment pins)
    depends_on("python@2.7,3.4:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-pip", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-numpy", type=("build", "run"))
    # katversion is used for versioning in setup.py but we'll override it via env var

    def setup_build_environment(self, env):
        # Ensure Spack provides dependencies without pip attempting isolation
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Provide deterministic version metadata for katversion
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_KATBEAM", self.spec.version.string)

    def test_katbeam_import(self):
        """Verify that katbeam can be imported and JimBeam class is accessible."""
        python = self.spec["python"].command
        code = (
            "import katbeam\n"
            "from katbeam import JimBeam\n"
            "beam = JimBeam('MKAT-AA-L-JIM-2020')\n"
            "print('KATBEAM_IMPORT_OK', beam.name)\n"
        )
        python("-c", code)

