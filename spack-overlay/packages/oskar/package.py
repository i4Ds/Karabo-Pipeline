from spack.package import *
import os


class Oskar(CMakePackage):
    """OSKAR is a software package for simulating radio interferometry observations."""

    homepage = "https://github.com/OxfordSKA/OSKAR"
    git = "https://github.com/OxfordSKA/OSKAR.git"

    license("BSD-3-Clause")
    maintainers("karabo")

    # Karabo uses 2.8.3 specifically
    # 2.10.0 works on arm64 but gives code -115 when reading vis files
    version("2.8.3", commit="2.8.3", preferred=True)
    version("2.10.0", commit="2.10.0")

    # Build dependencies
    depends_on("cmake@3.10:", type="build")
    depends_on("git", type="build")

    # Runtime dependencies
    depends_on("python@3.6:", type=("build", "run"))
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-cython", type="build")

    # Optional dependencies for enhanced functionality
    depends_on("hdf5+hl", type=("build", "run"))
    depends_on("cfitsio", type=("build", "run"))
    depends_on("fftw", type=("build", "run"))

    # Additional dependencies that may be needed for Python package
    depends_on("py-wheel", type="build")
    depends_on("py-pip", type="build")
    depends_on("py-build", type="build")
    depends_on("py-setuptools-scm", type="build")

    # Variants for optional features
    variant("cuda", default=False, description="Enable CUDA support")
    variant("openmp", default=True, description="Enable OpenMP support")

    # CUDA dependency when variant is enabled
    depends_on("cuda", when="+cuda")

    def cmake_args(self):
        """Configure CMake build arguments."""
        args = [
            self.define("CMAKE_BUILD_TYPE", "Release"),
            self.define("CMAKE_INSTALL_PREFIX", self.prefix),
        ]

        # OSKAR 2.8.3 doesn't use ENABLE_CUDA/ENABLE_OPENMP variables
        # These are handled automatically by CMake based on available libraries
        # Just ensure we have the right dependencies

        return args

    def setup_run_environment(self, env):
        """Set up environment variables for runtime."""
        # Set OSKAR include and library directories
        env.set("OSKAR_INC_DIR", self.prefix.include)
        env.set("OSKAR_LIB_DIR", self.prefix.lib)

        # Add OSKAR lib to LD_LIBRARY_PATH
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib64)

    def setup_dependent_run_environment(self, env, dependent_spec):
        """Set up environment variables for dependent packages."""
        env.set("OSKAR_INC_DIR", self.prefix.include)
        env.set("OSKAR_LIB_DIR", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib64)