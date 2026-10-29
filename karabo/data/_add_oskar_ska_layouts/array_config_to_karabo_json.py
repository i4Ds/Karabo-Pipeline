"""Converts an SKA-Mid array config into a Karabo telescope JSON with dish SEFDs.

Positions, names and diameters come from ska_ost_array_config, SEFDs per dish type
and receiver band from ska_ost_senscalc.
"""  # noqa: E501

import json
import os
import re
from importlib.metadata import version

import astropy.units as u
import numpy as np
import ska_ost_senscalc.mid_utilities as mid_utils
from ska_ost_array_config.array_config import MidSubArray
from ska_ost_senscalc.mid.sefd import SEFD_antenna
from ska_ost_senscalc.mid.validation import BAND_LIMITS
from ska_ost_senscalc.utilities import DishType, TelParams

SUBARRAY_TYPE = "AA*"
OUTPUT_FILE = (
    f"SKA-MID-AAstar.ska-ost-array-config-{version('ska-ost-array-config')}.json"
)

ELEVATION = 45 * u.deg
PWV_MM = 10
GALACTIC_ALPHA = 2.75
SAMPLES_PER_BAND = 5
# SKA-Mid receiver bands; Bands 3 and 4 are not part of the current SKA plans
BANDS = ["Band 1", "Band 2", "Band 5a", "Band 5b"]
# calculate_Trcv rejects SKA Band 5b above 15.3 GHz, although BAND_LIMITS allows 15.4
MAX_FREQUENCY_HZ = 15.3e9

DISH_TYPES = {
    "MID": (DishType.SKA1, "ska"),
    "MEERKAT": (DishType.MeerKAT, "meerkat"),
}

TEMP_CASA_FILE = "casa_antenna_list.txt"


def sefd_jy(freq: u.Quantity, band: str, dish_type: DishType) -> float:
    t_gal = 17.1 * u.K * (0.408 / freq.to_value(u.GHz)) ** GALACTIC_ALPHA
    t_sky = mid_utils.Tsky(t_gal, freq, ELEVATION, PWV_MM)
    t_sys = mid_utils.Tsys_dish(
        mid_utils.Trcv(freq, band, dish_type),
        mid_utils.Tspl(dish_type),
        t_sky,
        freq,
    )
    area = mid_utils.eta_dish(freq, dish_type) * TelParams.dish_area(dish_type)
    return float(np.squeeze(SEFD_antenna(t_sys, area).to_value(u.Jy)))


def sefd_curves(dish_type: DishType, limits_type: str) -> dict:
    curves = {}
    for band in BANDS:
        for entry in BAND_LIMITS[band]:
            if entry["type"] != limits_type:
                continue
            low, high = entry["limits"][0], min(entry["limits"][1], MAX_FREQUENCY_HZ)
            freqs_ghz = np.round(np.linspace(low, high, SAMPLES_PER_BAND) / 1e9, 6)
            curves[band] = [
                [float(round(f * 1e9)), round(sefd_jy(f * u.GHz, band, dish_type), 1)]
                for f in freqs_ghz
            ]
    return curves


array = MidSubArray(subarray_type=SUBARRAY_TYPE)
vp_types = dict(zip(array.array_config.names.data, array.array_config.vp_type.data))
array.generate_casa_antenna_list(TEMP_CASA_FILE)

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
            x, y, z, diameter, name = line.split("\t")[:5]
            stations.append((float(x), float(y), float(z), float(diameter), name))
assert long is not None
assert lat is not None
assert coordsys == "LOC"
assert len(stations) > 0
print(f"Parsed {len(stations)} stations")

curves = {
    vp_type: sefd_curves(dish_type, limits_type)
    for vp_type, (dish_type, limits_type) in DISH_TYPES.items()
}
telescope = {
    "format": "karabo-telescope",
    "version": 1,
    "name": "SKA-MID-AAstar",
    "centre": {"longitude_deg": long, "latitude_deg": lat, "altitude_m": 0.0},
    "sefd_reference": {
        "source": f"ska-ost-senscalc {version('ska-ost-senscalc')}",
        "convention": "per dish and per polarisation; excludes system "
        "efficiency and atmospheric attenuation",
        "elevation_deg": ELEVATION.to_value(u.deg),
        "pwv_mm": PWV_MM,
        "galactic_temperature": f"17.1 K (0.408 GHz / f)^{GALACTIC_ALPHA}",
    },
    "element_layouts": {"dish": [[0.0, 0.0, 0.0]]},
    "sefd_curves": curves,
    "stations": [
        {
            "name": name,
            "position_m": [x, y, z],
            "elements": "dish",
            "dish_diameter_m": diameter,
            "sefd": vp_types[name],
        }
        for x, y, z, diameter, name in stations
    ],
}

text = json.dumps(telescope, indent=2)
text = re.sub(
    r"\[\s+([-+0-9.eE,\s]+?)\s+\]",
    lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]",
    text,
)
with open(OUTPUT_FILE, "w") as f:
    f.write(text + "\n")
print(f"Wrote {OUTPUT_FILE}")

os.remove(TEMP_CASA_FILE)
