from spack.package import *


class PySkaSdpDatamodels(PythonPackage):
    """SKA SDP data models (Python)."""

    homepage = "https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels"
    git = "https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels.git"

    maintainers("karabo")
    license("BSD-3-Clause")

    version("0.1.3", tag="0.1.3")

    # Use pip/setuptools backends (Spack v0.23 does not support python_pep517)
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
    # Some tags still require poetry backend during metadata prep
    depends_on("py-poetry", type="build")

    # Runtime pins aligned with rascil.Dockerfile
    depends_on("py-numpy@1.23:", type=("build", "run"))
    depends_on("py-xarray@2022.12.0", type=("build", "run"))
    depends_on("py-astropy@5.1:", type=("build", "run"))
    depends_on("py-h5py@3.7:", type=("build", "run"))
    depends_on("py-pandas@1.5:", type=("build", "run"))

    import_modules = ["ska_sdp_datamodels"]

    def setup_build_environment(self, env):
        # Avoid SCM version resolution during build in minimal git context
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SKA_SDP_DATAMODELS", "0.1.3")
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION", "0.1.3")

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import ska_sdp_datamodels as s; print(getattr(s,'__version__','0')) ")


