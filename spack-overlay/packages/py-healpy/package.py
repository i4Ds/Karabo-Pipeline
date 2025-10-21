# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *
import os


class PyHealpy(PythonPackage):
    """healpy is a Python package to handle pixelated data on the sphere."""

    homepage = "https://healpy.readthedocs.io/"
    pypi = "healpy/healpy-1.13.0.tar.gz"

    license("GPL-2.0-only")

    version("1.17.3", sha256="4b9f6ae44c6a5a2922b6542b2086d53cc3a6b51543d856d18406fb984edbec5f")
    version("1.16.6", sha256="0ab26e828fcd251a141095af6d9bf3dba43cec6f0f5cd48b65bf0af8f56329f1")
    version("1.16.2", sha256="b7b9433152ff297f88fc5cc1277402a3346ff833e0fb7e026330dfac454de480")
    version("1.14.0", sha256="2720b5f96c314bdfdd20b6ffc0643ac8091faefcf8fd20a4083cedff85a66c5e")
    version("1.13.0", sha256="d0ae02791c2404002a09c643e9e50bc58e3d258f702c736dc1f39ce1e6526f73")
    version("1.7.4", sha256="3cca7ed7786ffcca70e2f39f58844667ffb8521180ac890d4da651b459f51442")

    # Build backend and helpers
    build_system("python_pip")
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-build", type="build")
    depends_on("py-pyproject-hooks", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-cython@0.29:0.29", type="build")
    depends_on("py-setuptools-scm@6:", type="build")
    depends_on("py-pkgconfig", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("py-extension-helpers@1:", type="build")
    depends_on("py-numpy@1.13:", type=("build", "run"))
    # Make SciPy optional to allow minimal builds for import verification
    variant("scipy", default=True, description="Enable SciPy runtime dependency")
    # Build against vendored HEALPix C++ in healpy source instead of external healpix-cxx
    variant(
        "internal-healpix",
        default=True,
        description="Use bundled HEALPix C++ from healpy source; drop external healpix-cxx",
    )
    # Allow toggling MPI/OpenMP through to healpix-cxx and libsharp
    # variant("mpi", default=True, description="Enable MPI via healpix-cxx/libsharp")
    # variant("openmp", default=True, description="Enable OpenMP via healpix-cxx/libsharp")
    depends_on("py-scipy@1.10.1:1.10", type=("build", "run"), when="+scipy")
    depends_on("py-astropy", type=("build", "run"))
    # Optional plotting support; avoid heavy GUI stack for headless verify builds
    variant("plot", default=True, description="Enable plotting support via matplotlib")
    depends_on("py-matplotlib", type=("build", "run"), when="+plot")
    depends_on("py-six", type=("build", "run"))

    # CFITSIO version constraints based on healpy version
    # healpy 1.18.x requires CFITSIO 4.5+ (4.6.2 for 1.18.1, 4.5.0 for 1.18.0)
    depends_on("cfitsio@4.5:", type=("build", "run"), when="@1.18:")
    # healpy 1.17.x requires CFITSIO 4.3+
    depends_on("cfitsio@4.3:", type=("build", "run"), when="@1.17:")
    # healpy 1.16.x requires CFITSIO 4.1+
    depends_on("cfitsio@4.1:", type=("build", "run"), when="@1.16:")
    depends_on("cfitsio@:3.47", type=("build", "run"), when="@:1.15")
    # healpy 1.15.x requires CFITSIO 4.0+ (for HEALPix 3.81 compatibility)
    # depends_on("cfitsio@4.0:", type=("build", "run"), when="@1.15:")
    # healpy 1.14.x requires CFITSIO 3.48+
    depends_on("cfitsio@3.48:", type=("build", "run"), when="@1.14:")
    # healpy 1.13.x and earlier can use CFITSIO 3.x
    depends_on("cfitsio@3.0:", type=("build", "run"), when="@:1.13")

    with when("~internal-healpix"):
        # HEALPix C++ version constraints based on healpy version
        # See HEALPY_HEALPIX_VERSION_MAPPING.md for detailed version mapping
        # healpy 1.18.x series requires HEALPix C++ 3.83
        depends_on("healpix-cxx@3.83:", type=("build", "run", "test"), when="@1.18:")
        # healpy 1.17.x series maps to HEALPix C++ 3.82
        depends_on("healpix-cxx@3.82:3.82", type=("build", "run", "test"), when="@1.17:")
        # healpy 1.16.x series requires HEALPix C++ ~3.82 (SVN r1206-1228)
        depends_on("healpix-cxx@3.80:3.82", type=("build", "run", "test"), when="@1.16:")
        # healpy 1.15.x series requires HEALPix C++ 3.81
        depends_on("healpix-cxx@3.81:3.81", type=("build", "run", "test"), when="@1.15:")
        # healpy 1.14.x series requires HEALPix C++ 3.70
        depends_on("healpix-cxx@3.70:3.79", type=("build", "run", "test"), when="@1.14:1.15")
        # healpy 1.13.x series requires HEALPix C++ 3.60
        depends_on("healpix-cxx@3.60:3.69", type=("build", "run", "test"), when="@1.13")
        # healpy 1.12.x series requires HEALPix C++ 3.50
        depends_on("healpix-cxx@3.50:3.59", type=("build", "run", "test"), when="@1.12")
        # healpy 1.8.x-1.11.x series requires HEALPix C++ 3.30
        depends_on("healpix-cxx@3.30:3.49", type=("build", "run", "test"), when="@1.8:1.11")
        # healpy 1.5.x-1.7.x series requires HEALPix C++ 3.11
        depends_on("healpix-cxx@3.11:3.29", type=("build", "run", "test"), when="@1.5:1.7")
        # healpy 1.3.x-1.4.x series requires HEALPix C++ ~3.0
        depends_on("healpix-cxx@3.0:3.10", type=("build", "run", "test"), when="@1.3:1.4")

        # Propagate variants to healpix-cxx when external is used
        # depends_on("healpix-cxx+mpi", when="+mpi")
        # depends_on("healpix-cxx~mpi", when="~mpi")
        # depends_on("healpix-cxx+openmp", when="+openmp")
        # depends_on("healpix-cxx~openmp", when="~openmp")

    depends_on("zlib", type=("build", "link"))
    depends_on("bzip2", type=("build", "link"))

    # When using vendored HEALPix C++, build the bundled libsharp copy.

    import_modules = ["healpy", "healpy._pixelfunc"]

    def patch(self):
        # Relax NumPy C-API deprecation guard in the build scripts so older
        # Cython-generated code can access legacy fields (e.g. dimensions).
        candidates = [
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
            "healpy/_setup_helpers.py",
        ]
        for rel in candidates:
            if os.path.exists(rel):
                ff = FileFilter(rel)
                ff.filter(
                    r"NPY_NO_DEPRECATED_API\s*[,=]\s*NPY_1_19_API_VERSION",
                    "NPY_NO_DEPRECATED_API=0",
                )
        # Create a forced-include header to override any upstream -DNPY_NO_DEPRECATED_API
        # definitions added by build tooling. The header is ensured to be included first
        # so our override wins regardless of command-line -D ordering.
        with open("spack_numpy_compat.h", "w", encoding="utf-8") as f:
            f.write(
                "#ifndef SPACK_NUMPY_COMPAT_HEADER\n"
                "#define SPACK_NUMPY_COMPAT_HEADER\n"
                "#undef NPY_NO_DEPRECATED_API\n"
                "#define NPY_NO_DEPRECATED_API 0\n"
                "#endif\n"
            )

    def setup_build_environment(self, env):
        # Ensure PEP517 builds use Spack-provided deps and do not create an
        # isolated environment that misses compiled libs like healpix-cxx.
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Provide version metadata when building from an sdist without VCS
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HEALPY", self.spec.version.string)
        # Healpy 1.16.x Cython output may access deprecated NumPy C-API fields
        # like PyArrayObject->dimensions; allow deprecated API to avoid build
        # failures with newer NumPy headers.
        # Force-include our compatibility header so it overrides any upstream
        # macro definitions.
        # Use absolute path to the header in the staged source directory.
        compat_hdr = os.path.join(self.stage.source_path, "spack_numpy_compat.h")
        env.append_flags("CFLAGS", f"-include {compat_hdr}")
        env.append_flags("CXXFLAGS", f"-include {compat_hdr}")
        # Also define explicitly as a fallback for sources compiled without the header
        env.append_flags("CFLAGS", "-DNPY_NO_DEPRECATED_API=0")
        env.append_flags("CXXFLAGS", "-DNPY_NO_DEPRECATED_API=0")

        # Building with +internal-healpix will build bundled libsharp.

    def test_import(self):
        python = which("python3") or which("python")
        if python:
            # Validate that the compiled extension is present
            code = (
                "import importlib, healpy; "
                "importlib.import_module('healpy._pixelfunc'); "
                "print(healpy.__version__)"
            )
            python("-c", code)
