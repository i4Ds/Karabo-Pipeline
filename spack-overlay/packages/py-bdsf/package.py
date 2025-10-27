# from https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/main/packages/py-bdsf/package.py
import os

from llnl.util import tty
from llnl.util.filesystem import working_dir

from spack.package import *
from spack.build_systems.python import PythonPipBuilder


class PyBdsf(PythonPackage):
    """PyBDSF: Blob Detection and Source Finder for radio interferometric images."""

    homepage = "https://github.com/lofar-astron/PyBDSF"
    git = "https://github.com/lofar-astron/PyBDSF.git"
    pypi = "bdsf/bdsf-1.0.0.tar.gz"

    version(
        "1.12.0",
        sha256="1ec301d7f98dd9dcc51245a793b63fa6a341f6378fea45907e06c6a453b6940a",
    )
    version(
        "1.11.0",
        sha256="975a58a5707456959ffff166bd20d0adeae7932262e31a6571c403cf36ab540f",
    )
    version(
        "1.10.2",
        sha256="eadde1f083647ebc611633f2b671f899f3baf96918fefc8c48df22eccc8d001b",
    )

    variant(
        "rap",
        default=False,
        description="Support reading 'rap' images using CasaCore.",
    )

    # Spack 0.23 compatible backends
    build_system("python_pip", "python_setuptools")

    # Python from https://github.com/lofar-astron/PyBDSF/releases
    depends_on('python@3.6:3.10', when='@:1.10', type=('build', 'run'))
    depends_on('python@3.8:3.12', when='@1.11:1.12', type=('build', 'run'))
    depends_on('python@3.13', when='@1.13:', type=('build', 'run'))

    # build deps from for tag in v1.10.2 v1.11.0 v1.12.0; do git show $tag:pyptoject.toml; done
    depends_on("py-setuptools@45:59", when="@:1.10", type="build")
    depends_on("py-setuptools@61:", when="@1.11:", type="build")
    # Avoid too-new setuptools breaking legacy builds; 70+ is fine for 1.12.0
    depends_on("py-setuptools@:70", when="@1.12.0:", type="build")

    # always
    depends_on("py-wheel", type="build")

    # PEP 517 builder and build tooling added from 1.11.0
    # scm8 needs minimum setuptools 61 (80 recommended)
    # scm7 needs setuptools 59-60
    # scm6 needs setuptools 45-58 (50 recommended)
    depends_on("py-build", when="@1.11:", type="build")
    depends_on("py-scikit-build@0.13:", when="@1.11:", type="build")
    # Spack may have an external cmake that is too old; allow alternatives
    depends_on("cmake@3.18:", when="@1.11:", type="build")
    depends_on("ninja", type="build")
    depends_on("py-setuptools-scm@8:", when="@1.11:", type="build")
    # depends_on("py-packaging", type="build")
    # depends_on("py-cython@0.29:", type="build", when="@:1.10")

    # Runtime deps
    # BDSF uses f2py from numpy at build time.
    depends_on("py-numpy@1.20:1", type=("build", "run"))
    depends_on("py-scipy@1.5:", type="run")
    depends_on("py-astropy@4:", type=("build", "run"))
    depends_on("py-matplotlib@3.3:", type="run")
    depends_on("py-casacore", type="run", when="+rap")
    # C++ extension links to Boost.Python and Boost.NumPy
    depends_on("boost+python+numpy")

    import_modules = ["bdsf"]

    def test_import(self):
        python = self.spec["python"].command
        if python:
            python("-c", "import bdsf; print('ok') ")

    def patch(self):
        # This fix ensures PyBDSF uses f2py from the $PATH (from numpy)
        # instead of /usr/bin/f2py3, which may not work.
        filter_file(
            "find_package(F2PY REQUIRED)",
            "set(F2PY_EXECUTABLE f2py)",
            "CMakeLists.txt",
            string=True,
        )

    def setup_build_environment(self, env):
        # Force pip to use Spack-provided setuptools/deps with correct constraints
        # Critical for scikit-build to find Boost and other Spack deps
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Make setuptools_scm generate a static version string when building
        # from sdists or VCS, avoiding any need for git metadata.
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BDSF", self.spec.version.string)
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION", self.spec.version.string)

        # Help CMake find Boost (headers and libs) and choose the correct
        # Boost.Python/Boost.NumPy component names for the active Python
        boost_spec = self.spec["boost"]

        self.stage.keep = True

        boost_include = boost_spec.headers.directories[0]
        env.prepend_path("CPATH", boost_include)

        for libdir in boost_spec.libs.directories:
            env.prepend_path("LIBRARY_PATH", libdir)
            env.prepend_path("LD_LIBRARY_PATH", libdir)

        pyver = self.spec["python"].version.up_to(2).string.replace(".", "")
        env.set("BOOST_PYTHON_LIB", f"boost_python{pyver}")
        env.set("BOOST_NUMPY_LIB", f"boost_numpy{pyver}")

        debug_pairs = []
        for dep, label in (
            ("py-setuptools", "py-setuptools"),
            ("py-setuptools-scm", "py-setuptools-scm"),
            ("py-scikit-build", "py-scikit-build"),
            ("py-scikit-build-core", "py-scikit-build-core"),
        ):
            try:
                debug_pairs.append(f"{label}={self.spec[dep].version.string}")
            except KeyError:
                # Dependency not active for this version slice
                continue

        if debug_pairs:
            debug_info = "\n".join(debug_pairs) + "\n"
            tty.warn("[py-bdsf] " + "; ".join(debug_pairs))
            env.set("BDSF_DEBUG_SETUPTOOLS", debug_info)
            try:
                with open(join_path(self.stage.path, "bdsf-setuptools-debug.txt"), "w") as fh:
                    fh.write(debug_info)
            except Exception:
                pass

    def install(self, spec, prefix):
        pip = spec["python"].command
        pip.add_default_arg("-m", "pip")

        args = PythonPipBuilder.std_args(self)
        log_file = join_path(self.stage.path, "pip-debug.log")
        # Always disable build isolation; we rely on Spack-provided
        # dependencies instead of network lookups inside pip.
        if "--no-build-isolation" not in args:
            args.append("--no-build-isolation")

        args.append(f"--prefix={prefix}")
        args.extend(["--log", log_file])
        args.append(".")

        with working_dir(self.stage.source_path):
            try:
                pip(*args)
            except ProcessError:
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r") as fh:
                            tail = fh.readlines()[-40:]
                        tty.error("[py-bdsf] pip tail:\n" + "".join(tail))
                    except Exception:
                        pass
                raise


