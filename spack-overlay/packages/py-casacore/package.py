from spack.package import *


class PyCasacore(PythonPackage):
    """Python bindings for casacore."""

    homepage = "https://github.com/casacore/python-casacore"
    git = "https://github.com/casacore/python-casacore.git"

    maintainers("karabo")
    license("GPL-2.0-or-later")

    version("3.5.0", tag="v3.5.0")

    # Build deps
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    depends_on("py-numpy@1.22:", type=("build", "run"))
    depends_on("py-six", type=("build", "run", "test"))

    # Link against casacore with Python enabled
    depends_on("casacore@3.5.0:+python")
    # Require Boost with Python and NumPy components for bindings
    depends_on("boost+python+numpy")

    # Environment for build to find casacore
    def setup_build_environment(self, env):
        env.prepend_path("CPATH", self.spec["casacore"].prefix.include)
        env.prepend_path("LIBRARY_PATH", join_path(self.spec["casacore"].prefix, "lib"))
        env.prepend_path("LIBRARY_PATH", join_path(self.spec["casacore"].prefix, "lib64"))
        env.prepend_path("LD_LIBRARY_PATH", join_path(self.spec["casacore"].prefix, "lib"))
        env.prepend_path("LD_LIBRARY_PATH", join_path(self.spec["casacore"].prefix, "lib64"))
        env.set("CASACORE_ROOT", self.spec["casacore"].prefix)
        # Help locate Boost.Python/Numpy libs and headers
        boost = self.spec["boost"].prefix
        env.set("BOOST_ROOT", boost)
        env.prepend_path("CPATH", join_path(boost, "include"))
        env.prepend_path("LIBRARY_PATH", join_path(boost, "lib"))
        env.prepend_path("LIBRARY_PATH", join_path(boost, "lib64"))
        env.prepend_path("LD_LIBRARY_PATH", join_path(boost, "lib"))
        env.prepend_path("LD_LIBRARY_PATH", join_path(boost, "lib64"))
        # Match Spack Boost Python/Numpy library naming: boost_python<pyXY>, boost_numpy<pyXY>
        pyver = self.spec["python"].version
        majmin = f"{pyver.up_to(2).string.replace('.', '')}"
        env.set("BOOST_PYTHON_LIB", f"boost_python{majmin}")
        env.set("BOOST_NUMPY_LIB", f"boost_numpy{majmin}")

    import_modules = ["casacore"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import casacore, casacore.tables, casacore.quanta; print('py-casacore OK') ")
