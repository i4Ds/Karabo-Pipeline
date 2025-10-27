"""Spack recipe for Karabo Pipeline.

This file uses Spack's DSL (version, depends_on, etc.), which confuses static
linters. Disable lints for this file.
"""  # flake8: noqa  # mypy: ignore-errors
# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportMissingModuleSource=false

from spack.package import version, build_system, PythonPackage, depends_on


class PyKarabo(PythonPackage):
    """A data-driven pipeline for Radio Astronomy from i4ds for the SKA Telescope.

    Karabo is a radio astronomy software distribution for validation and benchmarking
    of radio telescopes and algorithms. It can simulate the behavior of the Square Kilometer
    Array or other supported telescopes, includes OSKAR, RASCIL, WSClean, PyBDSF, and more.
    """

    homepage = "https://i4ds.github.io/Karabo-Pipeline/"
    git = "https://github.com/i4Ds/Karabo-Pipeline.git"

    # Version 0.34.0
    version("0.34.0", tag="v0.34.0", preferred=True)

    # Use pip-based build system with versioneer support
    build_system("python_pip")

    # Importable top-level module
    import_modules = ["karabo"]

    # Build dependencies
    depends_on("python@3.10:", type=("build", "run"))
    depends_on("py-setuptools@56:", type="build")
    depends_on("py-pip", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-versioneer+toml", type="build")

    # Core runtime dependencies matching environment.yaml
    depends_on("py-aratmospy@1.0.0", type=("build", "run"))
    depends_on("py-astropy", type=("build", "run"))
    depends_on("py-bdsf@1.10:", type=("build", "run"))
    depends_on("py-dask@2022.12.1", type=("build", "run"))
    depends_on("py-dask-mpi", type=("build", "run"))
    depends_on("py-distributed", type=("build", "run"))
    depends_on("py-eidos@1.1.0", type=("build", "run"))
    depends_on("py-healpy", type=("build", "run"))
    depends_on("py-h5py", type=("build", "run"))
    depends_on("py-katbeam@0.1.0", type=("build", "run"))
    depends_on("py-matplotlib", type=("build", "run"))
    depends_on("py-mpi4py", type=("build", "run"))
    depends_on("py-numpy@1.21:1.999", type=("build", "run"))
    depends_on("py-packaging", type=("build", "run"))
    depends_on("py-pandas", type=("build", "run"))
    depends_on("py-pyuvdata@2.4.2+casa", type=("build", "run"))
    depends_on("py-rascil@1.0.0", type=("build", "run"))
    depends_on("py-reproject@0.9:10.0", type=("build", "run"))
    depends_on("py-requests", type=("build", "run"))
    depends_on("py-rfc3986@2.0:", type=("build", "run"))
    depends_on("py-scipy@1.9:", type=("build", "run"))
    depends_on("py-ska-sdp-datamodels@0.1.3", type=("build", "run"))
    depends_on("py-ska-sdp-func-python@0.1.4", type=("build", "run"))
    depends_on(
        "py-xarray@2022.12.0:2023.2.0", type=("build", "run")
    )  # Match py-ska-sdp-datamodels constraint
    depends_on("wsclean", type=("build", "run"))
    depends_on("oskar", type=("build", "run"))

    def setup_build_environment(self, env):
        # Ensure Spack provides dependencies without pip attempting isolation
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Provide deterministic version metadata for versioneer
        env.set(
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_KARABO_PIPELINE",
            self.spec.version.string,
        )
        env.set("VERSIONEER_OVERRIDE", self.spec.version.string)

    def test_karabo_import(self):
        """Verify that Karabo can be imported and basic modules work."""
        python = self.spec["python"].command
        code = (
            "import karabo\n"
            "import karabo.simulation\n"
            "import karabo.imaging\n"
            "import karabo.sourcedetection\n"
            "print('KARABO_IMPORT_OK', karabo.__version__)\n"
        )
        python("-c", code)
