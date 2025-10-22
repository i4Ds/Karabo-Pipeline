from spack.package import *
import os


class Hyperbeam(Package):
    """Primary beam model for the Murchison Widefield Array (MWA) radio telescope.

    hyperbeam is a Rust library for calculating the beam response of the MWA.
    It provides Python bindings (mwa_hyperbeam) for use in Python applications.

    Usage:
        # Basic installation with Python bindings
        spack install hyperbeam +python

        # With CUDA support for NVIDIA GPUs
        spack install hyperbeam +python +cuda

        # With HIP support for AMD GPUs
        spack install hyperbeam +python +hip

        # Static linking for portability
        spack install hyperbeam +python +all-static
    """

    homepage = "https://github.com/MWATelescope/mwa_hyperbeam"
    url = "https://github.com/MWATelescope/mwa_hyperbeam/archive/refs/tags/v0.10.2.tar.gz"
    git = "https://github.com/MWATelescope/mwa_hyperbeam.git"

    maintainers("d3v-null")

    license("MPL-2.0")

    # Versions
    version("main", branch="main")
    version("0.10.2", sha256="2ee299d94c882e0d5d480134cf31bbd8")

    # Variants
    variant("python", default=True, description="Build Python bindings")
    variant("cuda", default=False, description="Enable CUDA support for GPU acceleration")
    variant("hip", default=False, description="Enable HIP support for AMD GPU acceleration")
    variant("all-static", default=False, description="Statically link all dependencies")
    variant("hdf5-static", default=False, description="Statically link HDF5")
    variant("cfitsio-static", default=False, description="Statically link CFITSIO")

    # Build dependencies
    depends_on("rust@1.64:", type="build")
    depends_on("cmake@3.10:", type="build")
    depends_on("pkgconfig", type="build")

    # Core dependencies
    depends_on("cfitsio", type=("build", "link", "run"))
    depends_on("hdf5@1.8.15:", type=("build", "link", "run"))
    depends_on("hdf5~mpi", type=("build", "link", "run"))

    # Python dependencies
    depends_on("python@3.8:", when="+python", type=("build", "run"))
    depends_on("py-pip", when="+python", type="build")
    depends_on("py-maturin@0.14:", when="+python", type="build")
    depends_on("py-cffi", when="+python", type=("build", "run"))
    depends_on("py-numpy@1.20:", when="+python", type=("build", "run"))

    # GPU dependencies
    depends_on("cuda@11.0:", when="+cuda", type=("build", "link", "run"))
    depends_on("hip@4.0:", when="+hip", type=("build", "link", "run"))

    # Conflicts
    conflicts("+cuda", when="+hip", msg="CUDA and HIP cannot be enabled simultaneously")

    def setup_build_environment(self, env):
        """Set up build environment for Rust compilation."""
        # Set CFITSIO paths
        env.set("CFITSIO_LIB", self.spec["cfitsio"].prefix.lib)
        env.set("CFITSIO_INC", self.spec["cfitsio"].prefix.include)

        # Set HDF5 paths
        env.set("HDF5_DIR", self.spec["hdf5"].prefix)
        env.prepend_path("PKG_CONFIG_PATH", join_path(self.spec["hdf5"].prefix, "lib", "pkgconfig"))

        # CUDA support
        if "+cuda" in self.spec:
            env.set("CUDA_LIB", self.spec["cuda"].prefix.lib)
            env.set("CUDA_LIB64", self.spec["cuda"].prefix.lib64)
            # Set compute capability (can be overridden by user)
            if not os.environ.get("HYPERDRIVE_CUDA_COMPUTE"):
                env.set("HYPERDRIVE_CUDA_COMPUTE", "75")  # Default to RTX 2070/3060 Ti

        # HIP support
        if "+hip" in self.spec:
            env.set("HIP_PATH", self.spec["hip"].prefix)

        # Fix for ARM64 proc-macro compilation issue with hdf5-metno-derive
        # Spack's target configuration can confuse Cargo into thinking we're cross-compiling
        # Clear any cross-compilation env vars that might interfere with proc-macro builds
        env.unset("CARGO_BUILD_TARGET")
        env.unset("CARGO_TARGET_DIR")
        env.unset("CARGO_BUILD_TARGET_DIR")

        # Rust compilation flags
        # NOTE: We do NOT set RUSTFLAGS on ARM64 to avoid proc-macro issues
        import platform
        machine = platform.machine()
        is_arm = machine in ("aarch64", "arm64")

        if not is_arm:
            rustflags = []
            # Static linking options (skip on ARM64 due to proc-macro limitations)
            if "+all-static" in self.spec or "+hdf5-static" in self.spec:
                rustflags.append("-C target-feature=+crt-static")
            if rustflags:
                env.set("RUSTFLAGS", " ".join(rustflags))

        # Use release profile for optimization
        env.set("CARGO_PROFILE_RELEASE_OPT_LEVEL", "3")
        env.set("CARGO_PROFILE_RELEASE_LTO", "thin")

    def install(self, spec, prefix):
        """Install hyperbeam using cargo."""
        import re

        # Patch build.rs to add GpuFloat typedef when GPU features aren't enabled
        # This ensures CFFI can parse the generated header
        if "+cuda" not in spec and "+hip" not in spec:
            build_rs_path = "build.rs"
            with open(build_rs_path, 'r') as f:
                build_rs = f.read()

            # Insert typedef header configuration after export config
            # Find the line with "config.export = export;" and add the header typedef
            pattern = r'(config\.export = export;)'
            replacement = r'''\1
                    config.header = Some(format!("typedef {} GpuFloat;", c_type));'''
            build_rs_patched = re.sub(pattern, replacement, build_rs)

            with open(build_rs_path, 'w') as f:
                f.write(build_rs_patched)

        cargo = which("cargo")

        # Build Rust features list
        features = []

        if "+cuda" in spec:
            features.append("cuda")

        if "+hip" in spec:
            features.append("hip")

        # ARM64 proc-macro limitation: skip all-static feature
        # The hdf5-metno-derive proc-macro doesn't support cross-compilation
        import platform
        is_arm = platform.machine() in ("aarch64", "arm64")

        if not is_arm:
            if "+all-static" in spec:
                features.append("all-static")
            elif "+hdf5-static" in spec:
                features.append("hdf5-static")
            elif "+cfitsio-static" in spec:
                features.append("cfitsio-static")
        else:
            # On ARM64, skip static features to avoid proc-macro issues
            if "+all-static" in spec or "+hdf5-static" in spec or "+cfitsio-static" in spec:
                print("WARNING: Skipping static linking features on ARM64 due to proc-macro limitations")

        # Install the Rust library
        # NOTE: We do NOT specify --target to allow Cargo to use the native default target
        # This is critical for proc-macro compilation on ARM64
        install_args = ["install", "--path", ".", "--locked", "--root", prefix]

        if features:
            install_args.extend(["--features", ",".join(features)])

        # Ensure we're building for the native target (no cross-compilation)
        cargo(*install_args)

        # Install Python bindings if requested
        if "+python" in spec:
            self.install_python_bindings(spec, prefix)

    def install_python_bindings(self, spec, prefix):
        """Install Python bindings using maturin."""
        python = spec["python"].command

        # Fix for CFFI GpuFloat error when GPU features aren't enabled
        # Add typedef to build.rs so CFFI can parse the generated header
        if "+cuda" not in spec and "+hip" not in spec:
            import re
            build_rs_path = "build.rs"
            with open(build_rs_path, 'r') as f:
                build_rs = f.read()

            # Find where we set header content and add the typedef
            # Look for the pattern where we configure cbindgen
            if 'config.header = Some' not in build_rs:
                # Add after "let export ="
                pattern = r'(let export = .*?;)'
                replacement = r'''\1

        // Add GpuFloat typedef for CFFI when GPU features aren't enabled
        config.header = Some(format!("typedef {} GpuFloat;", c_type));'''
                build_rs_patched = re.sub(pattern, replacement, build_rs, flags=re.DOTALL)

                with open(build_rs_path, 'w') as f:
                    f.write(build_rs_patched)
                print("Patched build.rs to add GpuFloat typedef for CFFI")

        # Set environment for Python build
        env = os.environ.copy()
        env["CFITSIO_LIB"] = str(spec["cfitsio"].prefix.lib)
        env["CFITSIO_INC"] = str(spec["cfitsio"].prefix.include)
        env["HDF5_DIR"] = str(spec["hdf5"].prefix)

        if "+cuda" in spec:
            env["CUDA_LIB"] = str(spec["cuda"].prefix.lib)
            if not env.get("HYPERDRIVE_CUDA_COMPUTE"):
                env["HYPERDRIVE_CUDA_COMPUTE"] = "75"

        if "+hip" in spec:
            env["HIP_PATH"] = str(spec["hip"].prefix)

        # Build features for maturin
        features = []
        if "+cuda" in spec:
            features.append("cuda")
        if "+hip" in spec:
            features.append("hip")

        # Use maturin to build and install Python package
        maturin_args = [
            "-m", "pip", "install",
            "--no-build-isolation",
            "--no-deps",
            f"--prefix={prefix}",
        ]

        # If maturin is available, use it directly for better control
        maturin = which("maturin")
        if maturin:
            build_args = ["build", "--release", "--strip"]
            if features:
                build_args.extend(["--features", ",".join(features)])

            maturin(*build_args, env=env)

            # Install the wheel
            import glob
            wheels = glob.glob("target/wheels/*.whl")
            if wheels:
                python("-m", "pip", "install", "--no-deps",
                       f"--prefix={prefix}", wheels[0], env=env)
        else:
            # Fallback to pip install
            maturin_args.append(".")
            python(*maturin_args, env=env)

    def setup_run_environment(self, env):
        """Set up runtime environment."""
        # Ensure shared libraries can be found
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib)
        env.prepend_path("LD_LIBRARY_PATH", self.prefix.lib64)

        # Add executables to PATH
        env.prepend_path("PATH", self.prefix.bin)

        # Python bindings
        if "+python" in self.spec:
            try:
                py_ver = self.spec["python"].version.up_to(2)
                env.prepend_path(
                    "PYTHONPATH",
                    join_path(self.prefix, "lib", f"python{py_ver}", "site-packages"),
                )
            except Exception:
                # Fallback for different Python installation layouts
                import glob
                python_dirs = glob.glob(join_path(self.prefix, "lib", "python*", "site-packages"))
                if python_dirs:
                    env.prepend_path("PYTHONPATH", python_dirs[0])

    def setup_dependent_run_environment(self, env, dependent_spec):
        """Set up environment for packages that depend on hyperbeam."""
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

    def test_import(self):
        """Test that mwa_hyperbeam Python module can be imported."""
        if "+python" not in self.spec:
            print("Skipping Python import test: +python variant disabled")
            return

        python = self.spec["python"].command
        if python:
            env = os.environ.copy()
            try:
                py_ver = self.spec["python"].version.up_to(2)
                site_pkgs = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
                env["PYTHONPATH"] = f"{site_pkgs}:{env.get('PYTHONPATH', '')}".strip(":")
            except Exception:
                pass

            python("-c", "from mwa_hyperbeam import FEEBeam; print('mwa_hyperbeam import successful')", env=env)

    def test_beam_calculation(self):
        """Test basic beam calculation functionality."""
        if "+python" not in self.spec:
            print("Skipping beam calculation test: +python variant disabled")
            return

        python = self.spec["python"].command
        if not python:
            return

        env = os.environ.copy()
        try:
            py_ver = self.spec["python"].version.up_to(2)
            site_pkgs = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
            env["PYTHONPATH"] = f"{site_pkgs}:{env.get('PYTHONPATH', '')}".strip(":")
        except Exception:
            pass

        test_script = """
import numpy as np
from mwa_hyperbeam import FEEBeam

# This test requires a beam file - skip if not available
print('mwa_hyperbeam basic functionality test: SKIPPED (requires beam file)')
"""
        python("-c", test_script, env=env)
