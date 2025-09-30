# /opt/spack/var/spack/repos/builtin/packages/py-astropy/package.py
# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

from spack.package import *


class PyAstropy(PythonPackage):
    """The Astropy Project is a community effort to develop a single core
    package for Astronomy in Python and foster interoperability between
    Python astronomy packages."""

    homepage = "https://astropy.org/"
    pypi = "astropy/astropy-4.0.1.post1.tar.gz"
    git = "https://github.com/astropy/astropy.git"

    import_modules = ["astropy"]

    license("BSD-3-Clause")

    version("6.1.0", sha256="6c3b915f10b1576190730ddce45f6245f9927dda3de6e3f692db45779708950f")
    version("5.1.1", sha256="ba4bd696af7090fd399b464c704bf27b5633121e461785edc70432606a94bd81")
    version("5.1", sha256="1db1b2c7eddfc773ca66fa33bd07b25d5b9c3b5eee2b934e0ca277fa5b1b7b7e")
    version(
        "4.0.1.post1", sha256="5c304a6c1845ca426e7bc319412b0363fccb4928cb4ba59298acd1918eec44b5"
    )
    version("3.2.1", sha256="706c0457789c78285e5464a5a336f5f0b058d646d60f4e5f5ba1f7d5bf424b28")
    version("2.0.14", sha256="618807068609a4d8aeb403a07624e9984f566adc0dc0f5d6b477c3658f31aeb6")
    version("1.1.2", sha256="6f0d84cd7dfb304bb437dda666406a1d42208c16204043bc920308ff8ffdfad1")
    version("1.1.post1", sha256="64427ec132620aeb038e4d8df94d6c30df4cc8b1c42a6d8c5b09907a31566a21")

    depends_on("c", type="build")  # generated

    variant("all", default=False, when="@3.2:", description="Enable all functionality")
    variant("recommended", default=True, when="@3.2:", description="Enable recommended functionality")

    # from analysis of pyproject.toml and setup.{py,cfg}

    # Required dependencies
    depends_on("python@3.10:", when="@6.1.0:", type=("build", "run"))
    depends_on("python@3.8:", when="@5.1:", type=("build", "run"))
    depends_on("py-setuptools@45:59", type="build")
    depends_on("py-cython@0.21:", when="@:2", type="build")
    depends_on("py-cython@0.29.13:", when="@:4", type="build")
    depends_on("py-cython@0.29.30:", when="@:5", type="build")
    depends_on("py-cython@3.0:3.0", when="@6:", type="build")

    # in newer pip versions --install-option does not exist
    # Newer pip dropped legacy --install-option; avoid breakage by pinning
    depends_on("py-pip@:23.2", type="build")

    depends_on("py-astropy-iers-data", when="@6:", type=("build", "run"))
    depends_on("py-numpy@2:", when="@6.1:", type=("build", "run"))
    depends_on("py-numpy@1.18:", when="@5.1:", type=("build", "run"))
    depends_on("py-numpy@1.16:", when="@4.0:", type=("build", "run"))
    depends_on("py-numpy@1.13:", when="@3.1:", type=("build", "run"))
    depends_on("py-numpy@1.10:", when="@3.0:", type=("build", "run"))
    depends_on("py-numpy@1.9:", when="@2.0:", type=("build", "run"))
    depends_on("py-numpy@1.7:", when="@1.2:", type=("build", "run"))
    depends_on("py-numpy@1.6:", type=("build", "run"))
    # https://github.com/astropy/astropy/issues/16200
    depends_on("py-numpy@:1", when="@:6.0")
    depends_on("py-packaging@19.0:", when="@5.1:", type=("build", "run"))
    depends_on("py-pyyaml@3.13:", when="@5.1:", type=("build", "run"))
    depends_on("py-pyerfa@2.0.0:2.0", when="@5.1:", type=("build", "run", "test"))
    depends_on("py-pyerfa@2.0.1.1:2.0.1", when="@6.1.0:", type=("build", "run", "test"))
    depends_on("py-setuptools-scm@6.2:6", when="@5.1:", type="build")
    depends_on("py-extension-helpers@1:1", when="@5.1:", type="build")
    depends_on("pkgconfig", type="build")

    depends_on("py-astropy-helpers", when="@:4", type="build")
    depends_on("py-jinja2@2.7:", when="@:4", type="build")

    # recommended dependencies
    with when("+recommended"):
        depends_on("py-scipy@1.8:", when="@6:", type=("build", "run"))
        depends_on("py-scipy@1.3:", when="@5:", type=("build", "run"))
        depends_on("py-scipy@0.18:", type=("build", "run"))
        depends_on("py-matplotlib@3.3:", when="@6:", type=("build", "run"))
        depends_on("py-matplotlib@3.1:", when="@5:", type=("build", "run"))
        depends_on("py-matplotlib@2.1:", when="@4:", type=("build", "run"))

    # Optional dependencies
    with when("+all"):
        depends_on("py-scipy@1.8:", when="@6:", type=("build", "run"))
        depends_on("py-scipy@1.3:", when="@5:", type=("build", "run"))
        depends_on("py-scipy@0.18:", type=("build", "run"))
        depends_on("py-matplotlib@3.3:", when="@6:", type=("build", "run"))
        depends_on("py-matplotlib@3.1:", when="@5:", type=("build", "run"))
        depends_on("py-matplotlib@2.1:", when="@4:", type=("build", "run"))
        depends_on("py-matplotlib@2.0:", type=("build", "run"))
        depends_on("py-certifi", when="@4.3:", type=("build", "run"))
        depends_on("py-dask+array", when="@4.1:", type=("build", "run"))
        depends_on("py-h5py", type=("build", "run"))
        depends_on("py-pyarrow@5:", when="@5:", type=("build", "run"))
        depends_on("py-beautifulsoup4", type=("build", "run"))
        depends_on("py-html5lib", type=("build", "run"))
        depends_on("py-bleach", type=("build", "run"))
        depends_on("py-pandas", type=("build", "run"))
        depends_on("py-sortedcontainers", type=("build", "run"))
        depends_on("py-pytz", type=("build", "run"))
        depends_on("py-jplephem", type=("build", "run"))
        depends_on("py-mpmath", type=("build", "run"))
        depends_on("py-asdf@2.10:", when="@5.1:", type=("build", "run"))
        depends_on("py-asdf@2.5:", when="@4.0.1post1:", type=("build", "run"))
        depends_on("py-asdf@2.3:", type=("build", "run"))
        depends_on("py-bottleneck", type=("build", "run"))
        depends_on("py-ipython@4.2:", when="@4.3:", type=("build", "run"))
        depends_on("py-ipython", type=("build", "run"))
        depends_on("py-pytest@7:", when="@5.0.2:", type=("build", "run"))
        depends_on("py-pytest", type=("build", "run"))
        depends_on("py-fsspec+http@2023.4:", when="@6.1:", type=("build", "run"))
        depends_on("py-s3fs@2023.4:", when="@6.1:", type=("build", "run"))
        depends_on("py-typing-extensions@3.10.0.1:", when="@5.0.2:", type=("build", "run"))

        # Historical optional dependencies
        depends_on("py-pyyaml", when="@:5", type=("build", "run"))
        depends_on("py-scikit-image", when="@:4.0", type=("build", "run"))
        depends_on("py-bintrees", when="@:3.2.1", type=("build", "run"))

        conflicts("^py-matplotlib@3.4.0,3.5.2")

    # System dependencies
    depends_on("erfa")
    depends_on("wcslib")
    depends_on("cfitsio@:3")
    depends_on("expat")

    def setup_build_environment(self, env):
        # Avoid PEP 517 build isolation fetching deps; Spack provides them.
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        # Ensure pyerfa is built from source within Spack, not as a wheel
        # that might not match ABI and would try to be pulled by pip.
        env.set("PIP_NO_BINARY", "pyerfa")
        # Provide deterministic version to setuptools_scm.
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ASTROPY", self.spec.version.string)

        ext_helpers_prefix = self.spec["py-extension-helpers"].prefix
        py_ver = self.spec["python"].version.up_to(2)
        site_dirs = []
        for libdir in ("lib", "lib64"):
            candidate = join_path(ext_helpers_prefix, libdir, f"python{py_ver}", "site-packages")
            if os.path.isdir(candidate):
                env.prepend_path("PYTHONPATH", candidate)
                site_dirs.append(candidate)
        setuptools_version = self.spec["py-setuptools"].version.string
        scm_version = self.spec["py-setuptools-scm"].version.string

        debug_info = (
            f"py-setuptools={setuptools_version}\n"
            f"py-setuptools-scm={scm_version}\n"
            f"extension_helpers_paths={':'.join(site_dirs) if site_dirs else '<none>'}\n"
        )

    def patch(self):
        # forces the rebuild of files with cython
        # avoids issues with PyCode_New() in newer
        # versions of python in the distributed
        # cython-ized files
        if os.path.exists("astropy/cython_version.py"):
            os.remove("astropy/cython_version.py")

    def install_options(self, spec, prefix):
        # For modern Astropy (5.x+ with pyproject), passing --install-option
        # flags via pip is not supported and causes failures. Rely on Spack's
        # dependency graph and environment variables set in setup_build_environment.
        if spec.satisfies("@5:"):
            return []
        # Legacy (pre-5.x) setup.py installs accept these options.
        return [
            "--use-system-libraries",
            "--use-system-erfa",
            "--use-system-wcslib",
            "--use-system-cfitsio",
            "--use-system-expat",
        ]

    @run_after("install")
    @on_package_attributes(run_tests=True)
    def install_test(self):
        with working_dir("spack-test", create=True):
            python("-c", "import astropy; astropy.test()")

    @property
    def skip_modules(self):
        modules = []

        if self.spec.satisfies("~extras"):
            modules.append("astropy.visualization.wcsaxes")

        return modules
