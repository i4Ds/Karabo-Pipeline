from spack.package import CMakePackage, depends_on, variant, version


class Wsclean(CMakePackage):
    """WSClean (w-stacking clean) is a fast generic widefield imager. It
    implements several gridding algorithms and offers fully-automated
    multi-scale multi-frequency deconvolution."""

    homepage = "https://wsclean.readthedocs.io/"
    git = "https://gitlab.com/aroffringa/wsclean.git"

    version("3.0", commit="v3.0", submodules=True)
    version("3.0.1", commit="v3.0.1", submodules=True)
    version("3.1", commit="v3.1", submodules=True)
    version("3.2", commit="v3.2", submodules=True)
    version("3.3", commit="v3.3", submodules=True)
    version("3.4", commit="v3.4", submodules=True)
    version("3.5", commit="v3.5", submodules=True)
    version(  # master branch at 2025-01-27
        "3.5.20250127",
        commit="d7d89fddf472ffac05a2fa1820ec7d0027da38ee",
        submodules=True,
    )
    version("3.6", commit="v3.6", submodules=True)
    version(  # master branch at 2025-06-30
        "3.6.20250630",
        commit="012778d4b9ca9a5e46815f7aae95f3f469539f99",
        submodules=True,
        preferred=True,
    )
    version(
        "latest", branch="master", submodules=True, no_cache=True, deprecated=True
    )
    version(
        "master",
        branch="master",
        submodules=True,
        no_cache=True,
    )

    variant("python", default=False, description="Enable Python support")
    variant("cuda", default=False, description="Enable CUDA support")
    # The 'mpi' flag forces using an MPI implementation from Spack, by adding
    # a dependency on it. When the flag is off, WSClean will still build
    # wsclean-mp if there's an MPI implementation outside the Spack tree.
    variant("mpi", default=True, description="Use MPI from Spack")

    depends_on("c", type="build", when="@:3.6.0")
    depends_on("cxx", type="build")

    # Linking WSClean against HDF5 fails if HDF5 was built with MPI support.
    depends_on("hdf5+cxx+threadsafe~mpi")
    depends_on("fftw")
    depends_on("casacore+data", when="@:3.5")
    depends_on("casacore@3.6:+data", when="@3.6:")
    depends_on("everybeam@0.2.0", when="@3.0")
    depends_on("everybeam@0.3.0", when="@3.0.1")
    depends_on("everybeam@0.3.0", when="@3.1")
    depends_on("everybeam@0.4.0", when="@3.2")
    depends_on("everybeam@0.5.1", when="@3.3")
    depends_on("everybeam@0.5.3", when="@3.4")
    depends_on("everybeam@0.6:0.7", when="@3.5")
    depends_on("everybeam@0.6:0.7", when="@3.6:3")
    depends_on("everybeam@0.6:", when="@latest,master")
    depends_on("idg@1.0.0", when="@3.1")
    depends_on("idg@1.1.0", when="@3.2")
    depends_on("idg@1.1.0", when="@3.3")
    depends_on("idg@1.2.0:", when="@3.4:")
    depends_on("idg+cuda", when="+cuda")
    depends_on("idg+python", when="+python")
    depends_on("boost+date_time+program_options")
    depends_on("openblas threads=pthreads")
    depends_on("gsl")
    depends_on("git")
    depends_on("python")
    depends_on("mpi", when="+mpi")

    def cmake_args(self):
        args = []

        # Disable MPI when ~mpi variant is used to prevent CMake from finding
        # stale system MPI configurations
        if "~mpi" in self.spec:
            args.append(self.define("CMAKE_DISABLE_FIND_PACKAGE_MPI", "ON"))

        return args

    def setup_build_environment(self, env):
        env.set("OPENBLAS_NUM_THREADS", "1")

    def setup_run_environment(self, env):
        env.set("OPENBLAS_NUM_THREADS", "1")
