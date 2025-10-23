from spack.package import *
import os
import shutil
import glob


class PyReproject(PythonPackage):
    """Astronomical image reprojection (astropy-affiliated)."""

    homepage = "https://github.com/astropy/reproject"
    git = "https://github.com/astropy/reproject.git"

    license("BSD-3-Clause")

    # Pin to 0.9.1 to satisfy RASCIL constraints
    version("0.9.1", tag="v0.9.1")
    version("0.14.1", tag="v0.14.1")

    # Use default PythonPackage build backend selection in this Spack version

    # Python and build deps
    depends_on("python@3.7:", type=("build", "run"))
    depends_on("py-setuptools@61:", type="build")
    depends_on("py-setuptools-scm", type="build")
    depends_on("py-wheel", type="build")
    depends_on("py-build", type="build")
    depends_on("py-pyproject-hooks", type="build")
    depends_on("py-pip", type="build")
    depends_on("py-cython@0.29:", type="build")
    depends_on("py-extension-helpers@1.0:", type="build")
    depends_on("py-numpy@1.18:", type=("build", "run"))
    depends_on("py-astropy@5:", type=("build", "run"))
    depends_on("py-scipy@1.5:", type=("build", "run"))
    depends_on("py-matplotlib@3.3:", type=("build", "run"))
    depends_on("py-pillow@8:", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    depends_on("py-astropy-healpix@1:", type=("build", "run"))

    import_modules = ["reproject", "reproject.reproject_interp"]

    def setup_build_environment(self, env):
        env.set("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_REPROJECT", self.spec.version.string)
        # Avoid PEP 517 creating an isolated env with mismatched setuptools
        env.set("PIP_NO_BUILD_ISOLATION", "1")
        env.set("PYTHONNOUSERSITE", "1")

    def test_import(self):
        python = self.spec["python"].command
        if python:
            python("-c", "import reproject; print(reproject.__version__) ")

    def test_healpix_api(self):
        python = self.spec["python"].command
        if python:
            code = (
                "from reproject.healpix import reproject_from_healpix; print('ok')"
            )
            python("-c", code)

    def test_top_level_interp(self):
        python = self.spec["python"].command
        if python:
            code = (
                "from reproject import reproject_interp; print('ok')"
            )
            python("-c", code)

    @run_after("install")
    def ensure_complete_package(self):
        """Ensure reproject has __init__.py and dist-info present in prefix.

        In older toolchains/views only compiled bits might be visible; copy
        missing Python shims from the staged source as a fallback.
        """
        py_ver = str(self.spec["python"].version.up_to(2))
        site_dir = join_path(self.prefix, "lib", f"python{py_ver}", "site-packages")
        pkg_dir = join_path(site_dir, "reproject")
        has_init = os.path.isfile(join_path(pkg_dir, "__init__.py"))
        has_dist = bool(glob.glob(join_path(site_dir, "reproject-*.dist-info")))
        if has_init and has_dist:
            return
        src_pkg = join_path(self.stage.source_path, "reproject")
        if os.path.isdir(src_pkg):
            mkdirp(pkg_dir)
            for name in os.listdir(src_pkg):
                if not name.endswith(".py"):
                    continue
                src = join_path(src_pkg, name)
                dst = join_path(pkg_dir, name)
                if not os.path.isfile(dst):
                    shutil.copyfile(src, dst)


