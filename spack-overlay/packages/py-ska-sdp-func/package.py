import llnl.util.lang

from spack.package import PythonPackage


class PySkaSdpFunc(PythonPackage):
    """
    Python bindings for SDP Processing Function Library, A collection
    of high-performance data processing utility functions for the
    Square Kilometre Array.
    """

    homepage = "https://gitlab.com/ska-telescope/sdp/ska-sdp-func"
    url = (
        "https://gitlab.com/ska-telescope/sdp/ska-sdp-func/"
        "-/archive/1.2.0/ska-sdp-func-1.2.0.tar.gz"
    )
    git = "https://gitlab.com/ska-telescope/sdp/ska-sdp-func"

    # maintainers("saliei")

    license("BSD-3-Clause")

    version("develop", branch="main")
    version(
        "1.2.2",
        sha256="7d40b3f8d0f18199a3ea85d4123af911a021a4e62a51140eac754c80f72a6c2c",
        preferred=True,
    )
    version(
        "1.2.1",
        sha256="ec1376d171f3130feb679fcad18d7783ce553fa9d75381ce7d4811a4005e98f3",
    )
    version(
        "1.2.0",
        sha256="4991003919aac8045b515cd9cd641d88fc1f886087e5d669f9e2d91b7e6d5b3d",
    )
    version(
        "1.1.7",
        sha256="b712499e9bf4b79c319b176de4450acfcd28c5edd2406bf8aac640f31db5e796",
    )
    version("0.0.6", branch="0.0.6-testing") # <- conda

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("cmake@3.1.0:", type="build")

    variant("cuda", default=False, description="Build CUDA kernels")
    variant(
        "cuda_arch",
        values=("6.0", "6.1", "6.2", "7.0", "7.5", "8.0", "8.6", "8.7"),
        default="all",
        multi=True,
        description="Build for CUDA arch",
    )
    variant("mkl", default=False, description="Build with Intel MKL support")

    depends_on("py-setuptools", type="build")
    depends_on("py-pytest", type="test")
    depends_on("py-numpy", type="run")

    depends_on("cuda@7.0.0:", when="+cuda")
    depends_on("intel-oneapi-mkl@2021.1.1:", when="+mkl")

    def setup_build_environment(self, env):
        cmake_args = []

        if "+cuda" in self.spec:
            cmake_args.append("-DFIND_CUDA=ON")
            if "+cuda_arch" in self.spec:
                cmake_args.append(
                    f"-DCUDA_ARCH={';'.join(self.spec['cuda_arch'].value)}"
                )
        else:
            cmake_args.append("-DFIND_CUDA=OFF")

        if "+mkl" in self.spec:
            cmake_args.append("-DFIND_MKL=ON")
        else:
            cmake_args.append("-DFIND_MKL=OFF")

        env.set("CMAKE_ARGS", " ".join(cmake_args))

    @property
    @llnl.util.lang.memoized
    def _output_version(self):
        spec_vers_str = str(self.spec.version.up_to(3))
        if "develop" in spec_vers_str:
            # Remove 'develop-' from the version in spack
            spec_vers_str = spec_vers_str.partition("-")[2]
        return spec_vers_str

    # Disabled - not sure where "pytest" would come from?
    # def test(self):
    #    pytest("-V")
