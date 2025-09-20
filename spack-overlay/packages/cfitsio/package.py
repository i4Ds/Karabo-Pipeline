from spack.package import *


class Cfitsio(AutotoolsPackage):
    """CFITSIO is a library of C and Fortran subroutines for reading and
    writing data files in FITS (Flexible Image Transport System) data format."""

    homepage = "https://heasarc.gsfc.nasa.gov/fitsio/"
    url = "https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3490.tar.gz"

    license("MIT")
    maintainers("karabo")

    # Match upstream 3.49; Spack tarball name variations are handled in builtin,
    # but we only need minimal metadata for our override.
    version("3.49", sha256="5b65a20d5c53494ec8f638267fca4a629836b7ac8dd0ef0266834eab270ed4b3")

    depends_on("bzip2")

    # Explicitly disable curl to avoid linking utils against libcurl, which has
    # caused undefined references in our environment.
    def configure_args(self):
        args = [
            "--disable-curl",
        ]
        return args



