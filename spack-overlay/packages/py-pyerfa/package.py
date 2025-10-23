from spack.package import *
import os
import shutil

class PyPyerfa(PythonPackage):
    """
    PyERFA is the Python wrapper for the ERFA library (Essential
    Routines for Fundamental Astronomy), a C library containing key
    algorithms for astronomy, which is based on the SOFA library
    published by the International Astronomical Union (IAU). All C
    routines are wrapped as Numpy universal functions, so that they
    can be called with scalar or array inputs.
    """

    homepage = "https://github.com/liberfa/pyerfa"
    pypi = "pyerfa/pyerfa-2.0.0.1.tar.gz"

    # Use modern PEP 517 build via pip backend
    build_system("python_pip")

    version("2.0.0.1", sha256="2fd4637ffe2c1e6ede7482c13f583ba7c73119d78bef90175448ce506a0ede30")
    version("2.0.1.5", sha256="17d6b24fe4846c65d5e7d8c362dcb08199dc63b30a236aedd73875cc83e1f6c0")

    depends_on("python@3.7:", type=("build", "run"))
    # Pin to numpy 1.23.5 to match our environment and avoid ABI issues
    depends_on("py-numpy@1.23.5:1", type=("build", "run"))
    depends_on("py-setuptools@61.2:", type="build", when="@2.0.1.5:")
    depends_on("py-setuptools@42:", type="build", when="@:2.0.0.1")
    depends_on("py-setuptools-scm@6.2:", type="build")
    depends_on("py-packaging", type=("build", "run"))
    depends_on("py-cython@0.29:", type="build")
    depends_on("py-pip@:25.2", type="build")
    depends_on("py-wheel", type="build")
    depends_on("erfa", type=("build", "link", "run"))
    depends_on("pkgconfig", type="build")
    depends_on("py-jinja2@2.10.3:", type="build", when="@:2.0.0.1")

    # Disable Spack's default import_module tests; we run our own below
    import_modules = ["erfa", "erfa.ufunc"]

    def setup_build_environment(self, env):
        """Set up build environment to find erfa and provide an SCM version."""
        spec = self.spec
        env.set("LDFLAGS", spec['erfa'].libs.ld_flags)
        env.set("CFLAGS", spec['erfa'].headers.include_flags)
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYERFA", spec.version.string)
        # Use default PEP517 build isolation to include all python modules

    @run_after("install")
    def ensure_upstream_erfa_python_package(self):
        """Ensure complete Python package presence alongside the compiled ext.

        Runtime vs. Test Environment mismatch
           - Spack runs Python package tests inside a temporary virtualenv
             distinct from the final, unified view. In that venv, pyerfa's
             module layout can differ slightly from what ultimately lands in the
             view (e.g., wheels vs. sdists, different backends).
           - Downstream packages (Astropy <=5.x, Healpy 1.16.x) expect the
             top-level module `erfa` to expose symbols such as ErfaError,
             ErfaWarning and constants like DAYSEC, DJY, ELG at import time.

        Observed failure modes
           - The final view occasionally contains only the compiled extension
             (e.g., `erfa/ufunc.*.so`) without all Python shim files
             (`__init__.py`, `version.py`, `core.py`), making
             `from erfa import ErfaError` or `erfa.DJY` fail at runtime.
        """
        py_ver = str(self.spec["python"].version.up_to(2))
        site_dir = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
        erfa_dir = join_path(site_dir, "erfa")
        upstream_erfa = join_path(self.stage.source_path, "erfa")
        if os.path.isdir(upstream_erfa):
            mkdirp(erfa_dir)
            # Copy all .py files from upstream erfa package, do not overwrite compiled .so
            for name in os.listdir(upstream_erfa):
                if name.endswith('.py'):
                    src = join_path(upstream_erfa, name)
                    dst = join_path(erfa_dir, name)
                    shutil.copyfile(src, dst)

    def test(self):
        """Self-contained test; works without a Spack view."""
        import os
        import subprocess

        python = self.spec["python"].command.path

        # Compute site-packages for the just-installed package and prepend to PYTHONPATH
        py_ver = str(self.spec["python"].version.up_to(2))
        site_dir = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{site_dir}:{env.get('PYTHONPATH','')}" if site_dir else env.get("PYTHONPATH", "")

        test_code = """
import sys
# Always ensure we can import the Python module installed by this package
try:
    import erfa
except Exception as exc:
    print(f"erfa missing or not importable: {exc}", file=sys.stderr)
    sys.exit(1)

# always ensure we can import these from erfa
try:
    from erfa import ErfaError, ErfaWarning, DAYSEC, DJY, ELG, __version__
except Exception as exc:
    print(f"erfa items missing or not importable: {exc}", file=sys.stderr)
    sys.exit(1)
"""
        subprocess.run([python, "-c", test_code], check=True, env=env)
