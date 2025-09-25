from spack.package import *


class PyBdsf(PythonPackage):
    """PyBDSF: Blob Detection and Source Finder for radio interferometric images."""

    homepage = "https://github.com/lofar-astron/PyBDSF"
    git = "https://github.com/lofar-astron/PyBDSF.git"

    maintainers("karabo")

    version("1.12.0", tag="v1.12.0")

    # Spack 0.23 compatible backends
    build_system("python_pip", "python_setuptools")

    # Python & build deps
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    depends_on("py-packaging", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-scikit-build", type="build")
    depends_on("py-cython@0.29:", type="build")

    # Runtime deps
    depends_on("py-numpy@1.20:", type=("build", "run"))
    depends_on("py-scipy@1.5:", type=("build", "run"))
    depends_on("py-astropy@4:", type=("build", "run"))
    # C++ extension links to Boost.Python and Boost.NumPy
    depends_on("boost+python+numpy")

    import_modules = ["bdsf"]

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            python("-c", "import bdsf; print('ok') ")

    def setup_build_environment(self, env):
        # Help CMake find Boost (headers and libs) and choose the correct
        # Boost.Python/Boost.NumPy component names for the active Python
        boost_prefix = self.spec["boost"].prefix
        env.set("BOOST_ROOT", boost_prefix)
        env.prepend_path("CPATH", join_path(boost_prefix, "include"))
        env.prepend_path("LIBRARY_PATH", join_path(boost_prefix, "lib"))
        env.prepend_path("LIBRARY_PATH", join_path(boost_prefix, "lib64"))
        env.prepend_path("LD_LIBRARY_PATH", join_path(boost_prefix, "lib"))
        env.prepend_path("LD_LIBRARY_PATH", join_path(boost_prefix, "lib64"))
        pyver = self.spec["python"].version
        majmin = f"{pyver.up_to(2).string.replace('.', '')}"
        env.set("BOOST_PYTHON_LIB", f"boost_python{majmin}")
        env.set("BOOST_NUMPY_LIB", f"boost_numpy{majmin}")


