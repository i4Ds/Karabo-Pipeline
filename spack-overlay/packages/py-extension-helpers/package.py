# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack.package import *


class PyExtensionHelpers(PythonPackage):
    """The extension-helpers package includes convenience helpers to
    assist with building Python packages with compiled C/Cython
    extensions. It is developed by the Astropy project but is intended
    to be general and usable by any Python package."""

    homepage = "https://github.com/astropy/astropy-helpers"
    pypi = "extension-helpers/extension-helpers-0.1.tar.gz"

    license("BSD-3-Clause")

    # Upstream retar/change on PyPI; update checksum to current blob
    version("1.1.1", sha256="f95dd304a523d4ff6680d9504fa1d68a4dd03bf3bfbbe0ade4d927ed9e693f00")

    build_system("python_pip")

    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-pip@21:", type="build")
    depends_on("py-setuptools@42:", type="build")
    depends_on("py-wheel", type="build")

    # Provide metadata when building from the PyPI sdist in an isolated tree
    def setup_build_environment(self, env):
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION", self.version.string)

