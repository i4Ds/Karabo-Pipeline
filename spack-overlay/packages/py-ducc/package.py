import os

from spack.package import *


class PyDucc(PythonPackage):
    """Python bindings for DUCC (ducc0)."""

    homepage = "https://gitlab.mpcdf.mpg.de/mtr/ducc/"
    pypi = "ducc0/ducc0-0.27.0.tar.gz"

    license("BSD-2-Clause")

    version("0.27.0", sha256="928a006712cd059c887647c5c42d145ddf409499d163be24c167ab4e828995b6")

    # Use default PythonPackage backend selection for this Spack version

    # Build deps
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools@69.2:69", type="build") # todo: constraints may be too tight
    depends_on("py-wheel@0.41.2:0.41", type="build") # todo: constraints may be too tight
    depends_on("py-build@1.2.1:1.2", type="build") # todo: constraints may be too tight
    depends_on("py-setuptools-scm@6.0.1:6", type="build") # todo: constraints may be too tight
    depends_on("py-pybind11@2.13.5:2.13", type="build") # todo: constraints may be too tight
    depends_on("py-packaging@24.1:24", type="build") # todo: constraints may be too tight
    depends_on("py-numpy@1.18:1", type=("build", "run")) # todo: constraints may be too tight

    import_modules = ["ducc0"]

    def setup_build_environment(self, env):
        # Force pip to use Spack-provided setuptools with correct constraints
        env.set("PIP_NO_BUILD_ISOLATION", "1")

    def patch(self):
        """Remove incomplete [project] table to let setuptools read setup.cfg."""
        pyproject_path = "pyproject.toml"

        if not os.path.exists(pyproject_path):
            return

        with open(pyproject_path, encoding="utf-8") as f:
            lines = f.readlines()

        try:
            start = lines.index("[project]\n")
        except ValueError:
            return

        end = start + 1
        while end < len(lines) and lines[end].strip():
            end += 1

        new_lines = lines[:start] + lines[end:]

        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    def test_import(self):
        python = self.spec["python"].command
        if python:
            python("-c", "import ducc0, sys; print(ducc0.__version__) ")


