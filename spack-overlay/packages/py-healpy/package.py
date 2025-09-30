# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class PyHealpy(PythonPackage):
    """healpy is a Python package to handle pixelated data on the sphere."""

    homepage = "https://healpy.readthedocs.io/"
    pypi = "healpy/healpy-1.13.0.tar.gz"

    license("GPL-2.0-only")

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
    depends_on("py-cython@0.29:", type="build")
    depends_on("py-setuptools-scm@6:", type="build")
    depends_on("py-pkgconfig", type="build")
    depends_on("pkgconfig", type="build")
    depends_on("py-extension-helpers@1:", type="build")
    depends_on("py-numpy@1.13:", type=("build", "run"))
    # Make SciPy optional to allow minimal builds for import verification
    variant("scipy", default=True, description="Enable SciPy runtime dependency")
    depends_on("py-scipy@1.10.1:1.10", type=("build", "run"), when="+scipy")
    depends_on("py-astropy", type=("build", "run"))
    # Optional plotting support; avoid heavy GUI stack for headless verify builds
    variant("plot", default=True, description="Enable plotting support via matplotlib")
    depends_on("py-matplotlib", type=("build", "run"), when="+plot")
    depends_on("py-six", type=("build", "run"))
    depends_on("cfitsio", type=("build", "run"))
    depends_on("healpix-cxx", type=("build", "run"))
    depends_on("zlib", type=("build", "link"))
    depends_on("bzip2", type=("build", "link"))

    import_modules = ["healpy", "healpy._pixelfunc"]

    def setup_build_environment(self, env):
        # Ensure PEP517 builds use Spack-provided deps and do not create an
        # isolated environment that misses compiled libs like healpix-cxx.
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Provide version metadata when building from an sdist without VCS
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HEALPY", self.spec.version.string)

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
