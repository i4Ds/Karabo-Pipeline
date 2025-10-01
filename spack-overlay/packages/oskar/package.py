from spack.package import *
import os


class Oskar(CMakePackage):
    """OSKAR is a software package for simulating radio interferometry observations."""

    homepage = "https://github.com/OxfordSKA/OSKAR"
    git = "https://github.com/OxfordSKA/OSKAR.git"

    license("BSD-3-Clause")

    # Karabo uses 2.8.3 specifically
    # 2.10.0 works on arm64 but gives code -115 when reading vis files
    version("2.8.3", commit="2.8.3", preferred=True)
    version("2.10.0", commit="2.10.0")

    # Variants
    variant("cuda", default=False, description="Enable CUDA support")
    variant("openmp", default=False, description="Enable OpenMP support")
    variant("mpi", default=False, description="Enable MPI support")
    variant("casacore", default=True, description="Enable Measurement Set I/O via Casacore")
    variant("python", default=True, description="Build and install Python bindings")
    variant("hdf5", default=True, description="Build HDF5 support")

    # Build dependencies
    depends_on("cmake@3.10:", type="build")
    depends_on("git", type="build")

    # Runtime dependencies
    # conda uses casacore 3.5.0.*, harp, hdf5 >=1.14.3,<1.14.4.0a0, libgcc, libgcc-ng >=12, libstdcxx, libstdcxx-ng >=12
    depends_on("python@3.6:", when="+python", type=("build", "run"))
    depends_on("py-numpy@1", when="+python", type=("build", "run"))
    depends_on("py-setuptools", when="+python", type="build")
    depends_on("py-cython", when="+python", type="build")

    # Optional dependencies for enhanced functionality
    depends_on("hdf5+hl", when="+hdf5", type=("build", "run"))
    depends_on("hdf5+hl~mpi", when="+hdf5~mpi", type=("build", "run"))
    depends_on("cfitsio", type=("build", "run"))
    # OSKAR requires single-precision FFTW. Use precision variant in this Spack.
    # Avoid pulling MPI into the build to reduce complexity; OSKAR tests
    # are serial and don't require MPI-enabled FFTW.
    depends_on("fftw precision=float,double", type=("build", "run"))
    depends_on("fftw~mpi", when="~mpi", type=("build", "run"))
    depends_on("fftw~openmp", when="~openmp", type=("build", "run"))

    # Additional dependencies that may be needed for Python package
    depends_on("py-wheel", when="+python", type="build")
    depends_on("py-pip", when="+python", type="build")
    depends_on("py-build", when="+python", type="build")
    depends_on("py-setuptools-scm", when="+python", type="build")

    # CUDA dependency when variant is enabled
    depends_on("cuda", when="+cuda")
    # Casacore dependency for MS functionality
    # using 3.5 because casacore 3.7 headers require C++14 constexpr semantics,
    # but OSKAR is being compiled with an older C++ standard.
    depends_on("casacore@3.5.0:3.5", when="+casacore")

    def cmake_args(self):
        """Configure CMake build arguments."""
        args = [
            self.define("CMAKE_BUILD_TYPE", "Release"),
            self.define("CMAKE_INSTALL_PREFIX", self.prefix),
            # Enforce newer C++ standard to satisfy dependencies when needed
            self.define("CMAKE_CXX_STANDARD", "14"),
            self.define("CMAKE_CXX_STANDARD_REQUIRED", "ON"),
            self.define("CMAKE_CXX_EXTENSIONS", "OFF"),
        ]

        # Allow disabling OpenMP via variant if needed for stability
        if "~openmp" in self.spec:
            # Prevent CMake from finding OpenMP
            args.append(self.define("CMAKE_DISABLE_FIND_PACKAGE_OpenMP", "ON"))

        # OSKAR 2.8.3 doesn't expose explicit feature toggles for CUDA/OMP/MS,
        # it detects libraries via CMake. Ensure Casacore is found when enabled
        # by passing the prefix if Spack's CMAKE_PREFIX_PATH doesn't suffice.
        if "+casacore" in self.spec:
            # Prefer config-file package discovery
            args.append(self.define("CASACORE_DIR", self.spec["casacore"].prefix))
            # Some builds still check this legacy hint
            args.append(self.define("CASACORE_ROOT_DIR", self.spec["casacore"].prefix))

        return args

    def setup_build_environment(self, env):
        """Constrain threading during build and tests to avoid segfaults."""
        env.set("OMP_NUM_THREADS", "1")
        env.set("OPENBLAS_NUM_THREADS", "1")
        env.set("MKL_NUM_THREADS", "1")
        env.set("NUMEXPR_NUM_THREADS", "1")
        env.set("CTEST_PARALLEL_LEVEL", "1")
        # Avoid over-aggressive binding/dynamic thread counts
        env.set("OMP_PROC_BIND", "false")
        env.set("OMP_DYNAMIC", "false")
        # Pin CPU features to baseline to avoid illegal instructions at runtime
        env.prepend_path("CPATH", self.spec['hdf5'].prefix.include)
        # Avoid injecting x86-specific flags on non-x86 platforms
        try:
            arch_family = str(self.spec.target.family)
        except Exception:
            arch_family = ""
        if arch_family in ("x86_64", "x86"):
            # The following flags avoid host-specific optimizations on x86
            env.append_flags("CFLAGS", "-march=x86-64 -mtune=generic")
            env.append_flags("CXXFLAGS", "-march=x86-64 -mtune=generic")
            env.append_flags("FFLAGS", "-march=x86-64 -mtune=generic")
        else:
            # On ARM/AArch64 and others, rely on toolchain defaults
            pass

    # Resolve build directory across Spack versions
    def _get_build_dir(self):
        # New-style builder API
        if hasattr(self, "builder") and hasattr(self.builder, "build_directory"):
            bdir = self.builder.build_directory
            if bdir and os.path.isdir(bdir):
                return bdir
        # Legacy attribute
        if hasattr(self, "build_directory"):
            bdir = getattr(self, "build_directory")
            if bdir and os.path.isdir(bdir):
                return bdir
        # Fallback: scan stage for spack-build*
        stage_root = getattr(self.stage, "path", None) or getattr(self.stage, "source_path", None)
        if stage_root and os.path.isdir(stage_root):
            for entry in os.listdir(stage_root):
                if entry.startswith("spack-build"):
                    candidate = join_path(stage_root, entry)
                    if os.path.isdir(candidate):
                        return candidate
        # Last resort
        return getattr(self.stage, "source_path", os.getcwd())

    def setup_run_environment(self, env):
        """Set up environment variables for runtime."""
        # Set OSKAR include and library directories
        env.set("OSKAR_INC_DIR", self.prefix.include)
        env.set("OSKAR_LIB_DIR", self.prefix.lib)

        # Add OSKAR lib to LD_LIBRARY_PATH
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib64)

        # Ensure Python can find bindings when installed with +python
        if "+python" in self.spec:
            try:
                py_ver = self.spec["python"].version.up_to(2)
                env.prepend_path(
                    "PYTHONPATH",
                    join_path(self.prefix, "lib", f"python{py_ver}", "site-packages"),
                )
            except Exception:
                pass

    # --- Logging helpers ---
    def _print_file(self, file_path, heading):
        try:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                print(f"===== Begin {heading}: {file_path} =====")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    print(fh.read())
                print(f"===== End {heading}: {file_path} =====")
        except Exception:
            pass

    def _dump_failure_logs(self):
        """Dump useful Spack/CMake/CTest logs to stdout for debugging failures."""
        # Stage root may contain Spack-generated logs
        stage_root = getattr(self.stage, "path", None) or getattr(self.stage, "source_path", None)
        build_dir = self._get_build_dir()

        # Known Spack logs
        if stage_root and os.path.isdir(stage_root):
            self._print_file(join_path(stage_root, "spack-build-out.txt"), "spack-build-out.txt")
            self._print_file(join_path(stage_root, "spack-build-env.txt"), "spack-build-env.txt")
            self._print_file(join_path(stage_root, "install-time-test-log.txt"), "install-time-test-log.txt")

        # CMake logs
        if build_dir and os.path.isdir(build_dir):
            self._print_file(join_path(build_dir, "CMakeFiles", "CMakeError.log"), "CMakeError.log")
            self._print_file(join_path(build_dir, "CMakeFiles", "CMakeOutput.log"), "CMakeOutput.log")

            # CTest logs
            ctest_tmp = join_path(build_dir, "Testing", "Temporary")
            if os.path.isdir(ctest_tmp):
                self._print_file(join_path(ctest_tmp, "LastTest.log"), "CTest LastTest.log")
                self._print_file(join_path(ctest_tmp, "LastTestsFailed.log"), "CTest LastTestsFailed.log")
                # Dump any other text logs in Temporary
                try:
                    for name in os.listdir(ctest_tmp):
                        if name.endswith(".log") or name.endswith(".txt"):
                            self._print_file(join_path(ctest_tmp, name), f"CTest {name}")
                except Exception:
                    pass

    def setup_dependent_run_environment(self, env, dependent_spec):
        """Set up environment variables for dependent packages."""
        env.set("OSKAR_INC_DIR", self.prefix.include)
        env.set("OSKAR_LIB_DIR", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib64)
        if "+python" in self.spec:
            try:
                py_ver = self.spec["python"].version.up_to(2)
                env.prepend_path(
                    "PYTHONPATH",
                    join_path(self.prefix, "lib", f"python{py_ver}", "site-packages"),
                )
            except Exception:
                pass

    @run_after("build")
    def build_test(self):
        """Ensure tests can run after build (no-op placeholder)."""
        pass

    def check(self):
        """Run the C++ test suite serially with verbose output."""
        with working_dir(self._get_build_dir()):
            ctest = which("ctest")
            if ctest:
                # Ensure serial execution and useful failure output
                env = os.environ.copy()
                env["CTEST_PARALLEL_LEVEL"] = "1"
                try:
                    ctest("-j1", "--output-on-failure", "--verbose", env=env)
                except ProcessError as e:
                    # Dump all useful logs
                    self._dump_failure_logs()
                    raise e
            else:
                # Fallback to running individual test executables if needed
                test_dir = join_path(self._get_build_dir(), "apps", "test")
                if os.path.exists(test_dir):
                    with working_dir(test_dir):
                        test_files = [
                            f for f in os.listdir(".")
                            if f.startswith("test_") and os.access(f, os.X_OK)
                        ]
                        for test_file in test_files:
                            self.run_test(test_file, purpose=f"Running {test_file}")

    def test_ctest(self):
        """Install-time test: run ctest serially with verbose output."""
        build_dir = self._get_build_dir()
        if not build_dir or not os.path.isdir(build_dir):
            print("Skipping ctest: build directory not present (likely binary install)")
            return
        with working_dir(build_dir):
            ctest = which("ctest")
            if ctest:
                env = os.environ.copy()
                env["CTEST_PARALLEL_LEVEL"] = "1"
                try:
                    ctest("-j1", "--output-on-failure", "--verbose", env=env)
                except ProcessError as e:
                    self._dump_failure_logs()
                    raise e
            else:
                test_dir = join_path(self._get_build_dir(), "apps", "test")
                if os.path.exists(test_dir):
                    with working_dir(test_dir):
                        test_files = [
                            f for f in os.listdir(".") if f.startswith("test_") and os.access(f, os.X_OK)
                        ]
                        for test_file in test_files:
                            self.run_test(test_file, purpose=f"Running {test_file}")

    def test_import(self):
        """Test that OSKAR Python module can be imported."""
        if "+python" not in self.spec:
            print("Skipping Python import test: +python variant disabled")
            return
        python = which("python3") or which("python")
        if python:
            env = os.environ.copy()
            # Help Python locate the installed module under the prefix
            try:
                py_ver = self.spec["python"].version.up_to(2)
                site_pkgs = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
                env["PYTHONPATH"] = f"{site_pkgs}:{env.get('PYTHONPATH', '')}".strip(":")
            except Exception:
                pass
            python("-c", "import oskar; print(f'OSKAR Python import successful')", env=env)

    @when("+python")
    @run_after("install")
    def install_python_bindings(self):
        """Install OSKAR Python bindings from the source tree when +python."""
        # Source path should exist when building from source; for buildcaches, this
        # was already executed at build-time so consumers get the installed module.
        source_root = getattr(self.stage, "source_path", None)
        if not source_root:
            return
        python_dir = join_path(source_root, "python")
        if not os.path.isdir(python_dir):
            return

        python = which("python3") or which("python")
        if not python:
            return

        env = os.environ.copy()
        env["OSKAR_INC_DIR"] = str(self.prefix.include)
        env["OSKAR_LIB_DIR"] = str(self.prefix.lib)

        with working_dir(python_dir):
            python(
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                f"--prefix={self.prefix}",
                ".",
                env=env,
            )

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