from spack.package import *


class PySkaSdpFunc(PythonPackage):
    """SKA SDP core functionality (Python module 'ska_sdp_func')."""

    homepage = "https://gitlab.com/ska-telescope/sdp/ska-sdp-func"
    git = "https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git"

    maintainers("karabo")

    # Pin to known working commit used previously in this stack
    version("0.1.0", commit="08eb17cf9f4d63320dd0618032ddabc6760188c9")

    # Spack 0.23-compatible backends
    build_system("python_pip", "python_setuptools")

    # Build dependencies
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    depends_on("py-packaging", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-scikit-build", type="build")
    depends_on("cmake@3.18:", type="build")

    # Runtime deps (align with our stack)
    depends_on("py-astropy@5.1:", type=("build", "run"))
    depends_on("py-numpy@1.23:", type=("build", "run"))
    depends_on("py-scipy@1.9:", type=("build", "run"))
    depends_on("py-ska-sdp-datamodels@0.1.3:", type=("build", "run"))

    import_modules = ["ska_sdp_func"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import ska_sdp_func as s; print('ok') ")


