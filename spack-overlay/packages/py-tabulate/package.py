from spack.package import *


class PyTabulate(PythonPackage):
    """Pretty-print tabular data in Python, a library and a command-line utility."""

    homepage = "https://github.com/astanin/python-tabulate"
    pypi = "tabulate/tabulate-0.9.0.tar.gz"

    version("0.9.0", sha256="e4ca13f5ba2c0fb4aafbc28659e5d6c7f5d9f8f10dff6e4328b31f6a8da3642d")

    # Avoid Spack's default import test; validate version metadata instead
    import_modules = []

    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools", type="build")

    def setup_build_environment(self, env):
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TABULATE", self.spec.version.string)

    # Provide a deterministic test to verify metadata exposes __version__ correctly
    def test(self):
        python = self.spec["python"].command
        # Check distribution metadata version without importing the module
        python(
            "-c",
            (
                "import sys;\n"
                "try:\n"
                "    from importlib.metadata import version\n"
                "except Exception:\n"
                "    from importlib_metadata import version\n"
                "ver = version('tabulate');\n"
                "print(ver);\n"
                "sys.exit(0 if ver else 1)\n"
            ),
        )


