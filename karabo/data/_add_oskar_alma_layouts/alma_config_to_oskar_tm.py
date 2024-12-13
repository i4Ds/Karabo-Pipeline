""" Gets the ALMA telescope configuration files for an observation cycle.
    The files are downloaded as an archive from the ALMA server. See
    constant CONFIG_FILE_URL for the URL. It converts the configuration
    files to an OSKAR telescope model directory. The folder are created
    in the current working directory.
    The script is based on the script 'array_config_to_oskar_tm.py'
    by M. Pluess, FHNW. You can find the code here:
    karabo/data/_add_oskar_ska_layouts.
"""

import glob
import math
import os
import shutil
import tarfile
import tempfile
from tarfile import TarFile
from zipfile import ZipFile

import requests

ALMA_CYCLE = 10
CONFIG_FILE_URL = f"https://almascience.eso.org/tools/documents-and-tools/cycle{ALMA_CYCLE}/alma-configuration-files"  # noqa: E501
CONFIG_FILE_BASENAME = f"alma-cycle{ALMA_CYCLE}-configurations"

# ALMA center coordinates
ALMA_CENTER_LON = -67.7538
ALMA_CENTER_LAT = -23.0234

# The config files are returned in an archive. Which one is in the header.
archive_type = {
    "application/zip": "zip",
    "application/x-tar": "tar",
    "application/gzip": "gz",
}


def convert_to_oskar_file(cfg_filename: str) -> str:
    """Convert a configuration file into an OSKAR tm-folder
    Args:
        cfg_filename: Full path to config-file (*.cfg)

    Raises:
        ValueError: In case the coordinate system in the file
            is not a local tangent plane (LOC)

    Returns:
        Full path to OSKAR output dir
    """
    coordsys = None
    stations = []
    with open(cfg_filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("# COFA="):
                long, lat = [float(s) for s in line.replace("# COFA=", "").split(",")]
            elif line.startswith("# coordsys="):
                # There may be a comment after coordinate system name
                coordsys = line.replace("# coordsys=", "").split(" ")[0]
            elif not line.startswith("#") and not line == "":
                x, y, z = line.split(" ")[:3]
                stations.append((x, y, z))

    if coordsys != "LOC":
        raise ValueError(
            f"Found coordinate system {coordsys} but expected"
            " LOC (Local tangent plane)."
            " This current coordinate system is not supported."
        )

    assert len(stations) > 0
    print(f"Parsed {len(stations)} stations")

    oskar_output_dir = f"{os.path.splitext(cfg_filename)[0]}.tm"
    os.makedirs(oskar_output_dir)

    with open(os.path.join(oskar_output_dir, "position.txt"), "w") as f:
        f.write(f"{ALMA_CENTER_LON} {ALMA_CENTER_LAT}\n")

    with open(os.path.join(oskar_output_dir, "layout.txt"), "w") as f:
        for x, y, z in stations:
            f.write(f"{x} {y} {z}\n")

    def get_nr_of_digits(i: int) -> int:
        return math.floor(math.log10(i) if i > 0 else 0.0) + 1

    target_nr_of_digits = get_nr_of_digits(len(stations))
    for station_nr in range(len(stations)):
        nr_of_digits = get_nr_of_digits(station_nr)
        station_nr_leading_zeros = "0" * (target_nr_of_digits - nr_of_digits) + str(
            station_nr
        )

        station_dir = os.path.join(
            oskar_output_dir, f"station{station_nr_leading_zeros}"
        )
        os.mkdir(station_dir)
        with open(os.path.join(station_dir, "layout.txt"), "w") as f:
            f.write("0.0 0.0\n")

    return oskar_output_dir


# Setup temp dir as cache and get the file
temp_dir = tempfile.mkdtemp()
response = requests.get(CONFIG_FILE_URL)

if response.status_code != 200:
    raise ConnectionError(
        "Could not download file."
        f" Server resoponded with status code {response.status_code}"
    )

# The config files come in different archives.
# Check the header to know which one.
file_extension = archive_type[response.headers["Content-Type"]]
if not file_extension:
    raise ValueError("Could not deduce file type from request header")

temp_filename = os.path.join(temp_dir, f"{CONFIG_FILE_BASENAME}.{file_extension}")

with open(temp_filename, "wb") as tmpfile:
    tmpfile.write(response.content)

if file_extension == "zip":
    with ZipFile(temp_filename) as configzip:
        configzip.extractall(path=temp_dir)
if file_extension == "tar":
    with TarFile(temp_filename) as configtar:
        configtar.extractall(path=temp_dir)
if file_extension == "gz":
    # This takes some extra work
    with tarfile.open(temp_filename, "r:gz") as tgz:
        tgz.extractall(path=temp_dir)
        # scandir return entries in arbitrary order. We must pich the folder
        with os.scandir(temp_dir) as it:
            for entry in it:
                if entry.is_dir():
                    temp_dir = os.path.join(temp_dir, entry)
                    break


cfg_files = glob.glob(os.path.join(temp_dir, "*.cfg"))
for file in cfg_files:
    tmp_folder = convert_to_oskar_file(file)
    shutil.copytree(tmp_folder, os.path.join(os.getcwd(), os.path.basename(tmp_folder)))

shutil.rmtree(temp_dir)
