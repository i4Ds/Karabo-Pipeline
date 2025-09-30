from spack.package import *


class PySkaSdpFuncPython(PythonPackage):
    """SKA SDP functionality (Python)."""

    homepage = "https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python"
    git = "https://gitlab.com/ska-telescope/sdp/ska-sdp-func-python.git"

    license("BSD-3-Clause")

    version("0.1.5", tag="0.1.5")
    version("0.1.4", tag="0.1.4") # <- conda

    build_system("python_pip", "python_setuptools")

    # Build deps
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    depends_on("py-packaging", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-hatchling", type="build")
    depends_on("py-hatch-vcs", type="build")
    depends_on("py-poetry", type="build")

    # Runtime pins (match rascil.Dockerfile stack)
    depends_on("py-astropy@5.1:", type=("build", "run"))
    depends_on("py-ducc@0.27:", type=("build", "run"))
    depends_on("py-numpy@1.23:", type=("build", "run"))
    depends_on("py-photutils@1.11:", type=("build", "run"))
    depends_on("py-scipy@1.9:", type=("build", "run"))
    depends_on("py-ska-sdp-datamodels@0.1.3:", type=("build", "run"))
    depends_on("py-xarray@2022.12.0:", type=("build", "run"))

    import_modules = ["ska_sdp_func_python"]

    def setup_build_environment(self, env):
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SKA_SDP_FUNC_PYTHON", self.spec.version.string)

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import ska_sdp_func_python as s; print('ok') ")


