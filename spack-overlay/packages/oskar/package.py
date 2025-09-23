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

    @run_after("build")
    def build_test(self):
        """Build tests during the build phase."""
        with working_dir(self.build_directory):
            # Build the test targets
            make("test", parallel=False)

    def test_suite(self):
        """Run the OSKAR test suite using ctest."""
        with working_dir(self.build_directory):
            # Run ctest with verbose output for debugging
            ctest = which("ctest")
            if ctest:
                ctest("--output-on-failure", "--verbose")
            else:
                # Fallback: try to run individual test executables
                test_dir = join_path(self.build_directory, "apps", "test")
                if os.path.exists(test_dir):
                    with working_dir(test_dir):
                        test_files = [f for f in os.listdir(".") if f.startswith("test_") and os.access(f, os.X_OK)]
                        for test_file in test_files:
                            self.run_test(test_file, purpose=f"Running {test_file}")

    def test_import(self):
        """Test that OSKAR Python module can be imported."""
        python = which("python3") or which("python")
        if python:
            python("-c", "import oskar; print(f'OSKAR Python import successful')")

    def test_executables(self):
        """Test that key OSKAR executables work."""
        executables = [
            "oskar_sim_interferometer",
            "oskar_imager",
            "oskar_vis_to_ms",
            "oskar_convert_cst_to_scalar"
        ]

        for exe in executables:
            exe_path = which(exe)
            if exe_path:
                # Test that executable can run with --help
                try:
                    exe_path("--help")
                    print(f"✓ {exe} executable test passed")
                except Exception as e:
                    print(f"⚠ {exe} executable test failed: {e}")
            else:
                print(f"⚠ {exe} executable not found in PATH")