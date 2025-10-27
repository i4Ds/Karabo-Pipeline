# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *
import os
import shutil
import glob


class PyPyerfa(PythonPackage):
    """PyERFA is the Python wrapper for the ERFA library (Essential
    Routines for Fundamental Astronomy), a C library containing key
    algorithms for astronomy, and is based on the SOFA library published
    by the International Astronomical Union (IAU).

    All C routines are wrapped as Numpy universal functions, so that
    they can be called with scalar or array inputs.
    """

    homepage = "https://github.com/liberfa/pyerfa"
    url = "https://github.com/liberfa/pyerfa/archive/v2.0.0.1.tar.gz"

    license("BSD-3-Clause")

    version("2.0.0.3", sha256="9c1458a30f7f265bd7fd5e214c7f683d8c99ea241bbf1fae1ab0be691d12da4b")
    version("2.0.0.1", sha256="2fd4637ffe2c1e6ede7482c13f583ba7c73119d78bef90175448ce506a0ede30")
    version("2.0.1.5", sha256="17d6b24fe4846c65d5e7d8c362dcb08199dc63b30a236aedd73875cc83e1f6c0")

    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-numpy@1.23.5:1", type=("build", "run"))
    depends_on("py-setuptools@61.2:69", type="build")
    depends_on("py-setuptools-scm@6:", type="build")  # Compatible with setuptools@45:
    depends_on("py-packaging", type=("build", "run"))
    depends_on("py-cython@0.29:", type="build")
    depends_on("py-wheel", type="build")  # Ensure wheel is available for proper install
    depends_on("erfa", type=("build", "link", "run"))
    depends_on("pkgconfig", type="build")
    depends_on("py-jinja2@2.10.3:", type="build", when="@:2.0.0.1")

    def setup_build_environment(self, env):
        spec = self.spec
        # Force pip to use Spack-provided setuptools with correct version constraints
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        env.set("LDFLAGS", spec['erfa'].libs.ld_flags)
        env.set("CFLAGS", spec['erfa'].headers.include_flags)
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYERFA", spec.version.string)

    @run_after("install")
    def ensure_complete_erfa_package(self):
        """Ensure erfa package has __init__.py and can be imported.

        With old setuptools + PIP_NO_BUILD_ISOLATION, pyerfa installs as
        a broken namespace package missing __init__.py and dist-info.
        """
        py_ver = str(self.spec["python"].version.up_to(2))
        site_dir = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
        erfa_dir = join_path(site_dir, "erfa")

        # Check if erfa package is complete
        has_init = os.path.isfile(join_path(erfa_dir, "__init__.py"))
        dist_infos = glob.glob(join_path(site_dir, "pyerfa-*.dist-info"))

        if has_init and dist_infos:
            return  # Package is complete

        # Package is incomplete - copy missing Python files from source
        upstream_erfa = join_path(self.stage.source_path, "erfa")
        if os.path.isdir(upstream_erfa):
            mkdirp(erfa_dir)
            for name in os.listdir(upstream_erfa):
                if not name.endswith(".py"):
                    continue
                src = join_path(upstream_erfa, name)
                dst = join_path(erfa_dir, name)
                if not os.path.isfile(dst):
                    shutil.copyfile(src, dst)
