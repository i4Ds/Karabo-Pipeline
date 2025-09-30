from spack.package import *


class PyPyprojectHooks(PythonPackage):
    """Wrappers to call pyproject.toml-based build backend hooks."""

    homepage = "https://github.com/pypa/pyproject-hooks"
    pypi = "pyproject_hooks/pyproject_hooks-1.0.0.tar.gz"

    version("1.0.0", sha256="f271b298b97f5955d53fb12b72c1fb1948c22c1a6b70b315c54cedaca0264ef5")

    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-flit-core@2:3", type="build")

    import_modules = ["pyproject_hooks"]
