from spack.package import *


class PyPyuvdata(PythonPackage):
    """An interface for reading, writing, and manipulating interferometric datasets in Python."""

    homepage = "https://github.com/RadioAstronomySoftwareGroup/pyuvdata"
    pypi = "pyuvdata/pyuvdata-2.4.2.tar.gz"

    maintainers("karabo")

    version("2.4.2", sha256="f5f3cfabf2a1f4a1ef0f0f3b41228d48a0a1bcb3f92b1b9c2a8217fac6d64e5f")

    # Runtime deps mirrored from upstream environment constraints
    depends_on("python@3.9:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-setuptools-scm@6:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    # Minimal runtime set needed for import
    depends_on("py-numpy@1.22:1.26", type=("build", "run"))
    depends_on("py-scipy@1.9:", type=("build", "run"))
    depends_on("py-astropy@5:", type=("build", "run"))
    depends_on("py-pyyaml@6:", type=("build", "run"))
    depends_on("py-h5py@3.7:", type=("build", "run"))
    # Extras used by some IO paths; keep optional to avoid hard requirements
    # Optional IO stack can be added by consumers if needed

    def setup_build_environment(self, env):
        # Avoid requiring git metadata during build
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION", str(self.version))

    # Ensure import works post-install
    def test(self):
        python = self.spec["python"].command
        python("-c", "import pyuvdata,sys; print(pyuvdata.__version__); sys.exit(0)")


