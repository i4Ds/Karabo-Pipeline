from spack.package import *


class PyXarray(PythonPackage):
    """N-D labeled arrays and datasets in Python"""

    homepage = "https://github.com/pydata/xarray"
    git = "https://github.com/pydata/xarray.git"

    # 'xarray.tests' requires 'pytest'. Leave out of 'import_modules' to avoid
    # unnecessary dependency.
    import_modules = [
        "xarray",
        "xarray.core",
        "xarray.plot",
        "xarray.util",
        "xarray.backends",
        "xarray.coding",
    ]

    license("Apache-2.0")
    maintainers("karabo")

    # Legacy version required by rascil==1.0.0 and ska-sdp-func-python==0.1.5
    version("2022.12.0", tag="v2022.12.0")

    # Variants similar to builtin for compatibility
    variant("io", default=False, description="Build io backends")
    variant("parallel", default=False, description="Build parallel backend")

    # Python and build deps
    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-setuptools-scm@3.4:+toml", type="build")

    # Runtime deps for 2022.12.0
    depends_on("py-numpy@1.23:1.23", when="@2022.12.0", type=("build", "run"))
    depends_on("py-pandas@1.5:", when="@2022.12.0", type=("build", "run"))
    depends_on("py-packaging@21.3:", when="@2022.12.0", type=("build", "run"))

    # Optional IO stack when +io
    depends_on("py-netcdf4", when="+io", type=("build", "run"))
    depends_on("py-h5netcdf", when="+io", type=("build", "run"))
    depends_on("py-scipy", when="+io", type=("build", "run"))
    depends_on("py-zarr", when="+io", type=("build", "run"))
    depends_on("py-fsspec", when="+io", type=("build", "run"))
    depends_on("py-cftime", when="+io", type=("build", "run"))


