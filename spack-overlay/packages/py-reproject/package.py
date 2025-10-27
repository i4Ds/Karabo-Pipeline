from spack.package import *


class PyReproject(PythonPackage):
    """Astronomical image reprojection (astropy-affiliated)."""

    homepage = "https://github.com/astropy/reproject"
    git = "https://github.com/astropy/reproject.git"

    license("BSD-3-Clause")

    # Pin to 0.9.1 to satisfy RASCIL constraints
    version("0.9.1", tag="v0.9.1")
    version("0.14.1", tag="v0.14.1")

    # Python and build deps
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools@61:69", type="build")
    depends_on("py-setuptools-scm@6:", type="build")  # Compatible with setuptools@45:
    depends_on("py-cython@0.29:", type="build")
    depends_on("py-extension-helpers@1.0:", type="build")
    depends_on("py-numpy@1.18:", type=("build", "run"))
    depends_on("py-astropy@5:", type=("build", "run"))
    depends_on("py-scipy@1.5:", type=("build", "run"))
    depends_on("py-matplotlib@3.3:", type=("build", "run"))
    depends_on("py-pillow@8:", type=("build", "run"))
    depends_on("py-astropy-healpix@1:", type=("build", "run"))

    def setup_build_environment(self, env):
        # Let pip's build isolation pick the ideal setuptools version
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_REPROJECT", self.spec.version.string)

    def test_import(self):
        python = self.spec["python"].command
        if python:
            python("-c", "import reproject; print(reproject.__version__)")

    def test_healpix_api(self):
        python = self.spec["python"].command
        if python:
            code = "from reproject.healpix import reproject_from_healpix; print('ok')"
            python("-c", code)

