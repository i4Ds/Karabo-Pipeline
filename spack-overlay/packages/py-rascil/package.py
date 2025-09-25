from spack.package import *


class PyRascil(PythonPackage):
    """Radio Astronomy Simulation, Calibration and Imaging Library (RASCIL)."""

    homepage = "https://gitlab.com/ska-telescope/sdp/rascil"
    # Avoid git (private); install from SKA artefact wheel via pip

    maintainers("karabo")

    # Prefer fetching a prebuilt wheel from SKA artefact to avoid git auth
    base_url = "https://artefact.skao.int/repository/pypi-all/packages/rascil/{ver}/rascil-{ver}-py3-none-any.whl"
    version("1.0.0")

    def url_for_version(self, version):
        ver = str(version)
        return self.base_url.format(ver=ver)

    # Spack 0.23-compatible backends
    build_system("python_pip", "python_setuptools")

    # Python runtime
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type=("build", "run", "test"))
    depends_on("py-wheel", type="build")

    # Direct RASCIL dependencies (match Docker ARG versions via environment)
    depends_on("py-astroplan@0.8:", type=("build", "run"))
    depends_on("py-astropy@5.1:", type=("build", "run"))
    depends_on("py-bdsf@1.12.0", type=("build", "run"))
    depends_on("py-casacore@3.5.0", type=("build", "run"))
    depends_on("py-dask-memusage", type=("build", "run"))
    depends_on("py-dask@2022.10.2", type=("build", "run"))
    depends_on("py-dask-memusage@1.1:", type=("build", "run"))
    depends_on("py-distributed@2022.10.2", type=("build", "run"))
    depends_on("py-matplotlib@3.6:", type=("build", "run"))
    depends_on("py-numpy@1.23:", type=("build", "run"))
    depends_on("py-pandas@1.5:", type=("build", "run"))
    depends_on("py-reproject@0.9:", type=("build", "run"))
    depends_on("py-scipy@1.9:", type=("build", "run"))
    depends_on("py-seqfile", type=("build", "run"))
    depends_on("py-ska-sdp-datamodels@0.1.3", type=("build", "run"))
    depends_on("py-ska-sdp-func-python@0.1.5", type=("build", "run"))
    depends_on("py-ska-sdp-func@0.1.0", type=("build", "run"))
    depends_on("py-tabulate", type=("build", "run"))
    depends_on("py-xarray@2022.12:", type=("build", "run"))

    # Transitively required C/C++ libs via other Python deps
    depends_on("casacore@3.5.0:+python")
    depends_on("cfitsio")
    depends_on("fftw")
    depends_on("hdf5@1.12:")

    import_modules = ["rascil"]

    def setup_build_environment(self, env):
        # Ensure pip can reach SKA artefact repo
        env.set("PIP_INDEX_URL", "https://artefact.skao.int/repository/pypi-all/simple")
        env.set("PIP_EXTRA_INDEX_URL", "https://pypi.org/simple")

    def install(self, spec, prefix):
        # Install directly from the staged wheel if present; otherwise fallback to name==version
        python = spec["python"].command
        archive = getattr(self.stage, "archive_file", None)
        if archive:
            python("-m", "pip", "install", "--no-build-isolation", "--no-deps", f"--prefix={prefix}", archive)
        else:
            python("-m", "pip", "install", "--no-build-isolation", "--no-deps", f"--prefix={prefix}", f"rascil=={self.version}")

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import rascil; print('py-rascil OK') ")


