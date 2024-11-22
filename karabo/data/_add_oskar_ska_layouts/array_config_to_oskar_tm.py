"""Converts an SKA array config from the package ska_ost_array_config to an OSKAR telescope model directory"""  # noqa: E501

import math
import os

import numpy as np
from ska_ost_array_config.array_config import LowSubArray

# from ska_ost_array_config.array_config import MidSubArray

OSKAR_OUTPUT_DIR = "SKA-LOW-AA4.ska-ost-array-config-2.3.1.tm"
ARRAY = LowSubArray(subarray_type="AA4")

TEMP_CASA_FILE = "casa_antenna_list.txt"

assert np.all(ARRAY.array_config.data_vars["offset"] == 0.0)
ARRAY.generate_casa_antenna_list(TEMP_CASA_FILE)

long = None
lat = None
coordsys = None
stations = []
with open(TEMP_CASA_FILE, "r") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("# COFA="):
            long, lat = [float(s) for s in line.replace("# COFA=", "").split(",")]
        elif line.startswith("# coordsys="):
            coordsys = line.replace("# coordsys=", "")
        elif not line.startswith("#") and not line == "":
            x, y, z = line.split("\t")[:3]
            stations.append((x, y, z))
assert long is not None
assert lat is not None
assert coordsys == "LOC"
assert len(stations) > 0
print(f"Parsed {len(stations)} stations")

os.makedirs(OSKAR_OUTPUT_DIR)

with open(os.path.join(OSKAR_OUTPUT_DIR, "position.txt"), "w") as f:
    f.write(f"{long} {lat}\n")

with open(os.path.join(OSKAR_OUTPUT_DIR, "layout.txt"), "w") as f:
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
    station_dir = os.path.join(OSKAR_OUTPUT_DIR, f"station{station_nr_leading_zeros}")
    os.mkdir(station_dir)
    with open(os.path.join(station_dir, "layout.txt"), "w") as f:
        f.write("0.0 0.0\n")

os.remove(TEMP_CASA_FILE)
