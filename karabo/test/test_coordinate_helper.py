import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from numpy.typing import NDArray

import astropy.units as u

from karabo.simulation.coordinate_helper import (
    east_north_to_long_lat,
    wgs84_to_cartesian,
)

# ---------------------------------------------------------------------------
# Reference telescope sites  (lon [deg], lat [deg], alt [m])
# ---------------------------------------------------------------------------
LOFAR = (6.86763008, 52.91139459, 50)
MEERKAT = (21.4430, -30.7130, 1054)
ASKAP = (116.6310, -26.6970, 377)
VLA = (-107.6184, 34.0784, 2124)

SITES = [LOFAR, MEERKAT, ASKAP, VLA]

# Tolerances
ECEF_ATOL_M = 1e-3  # 1 mm
GEODETIC_ATOL_DEG = 1e-10


# ===================================================================
# Helpers
# ===================================================================
def _astropy_ecef(lon_deg: float, lat_deg: float, alt_m: float) -> np.ndarray:
    """Return ECEF [x, y, z] in metres via astropy (ground truth)."""
    loc = EarthLocation.from_geodetic(
        lon=lon_deg * u.deg, lat=lat_deg * u.deg, height=alt_m * u.m
    )
    return np.array([loc.x.to(u.m).value, loc.y.to(u.m).value, loc.z.to(u.m).value])


def _astropy_enu_to_geodetic(
    east: float,
    north: float,
    lon0: float,
    lat0: float,
    alt0: float = 0.0,
    up: float = 0.0,
) -> tuple[float, float]:
    """Apply an ENU offset to a reference point and return (lon, lat) via astropy."""
    ref = EarthLocation.from_geodetic(
        lon=lon0 * u.deg, lat=lat0 * u.deg, height=alt0 * u.m
    )
    x0, y0, z0 = ref.x.to(u.m).value, ref.y.to(u.m).value, ref.z.to(u.m).value

    lon_r = np.deg2rad(lon0)
    lat_r = np.deg2rad(lat0)
    sl, cl = np.sin(lon_r), np.cos(lon_r)
    sp, cp = np.sin(lat_r), np.cos(lat_r)

    dx = -sl * east - sp * cl * north + cp * cl * up
    dy = cl * east - sp * sl * north + cp * sl * up
    dz = cp * north + sp * up

    target = EarthLocation.from_geocentric(
        x=(x0 + dx) * u.m, y=(y0 + dy) * u.m, z=(z0 + dz) * u.m
    )
    return target.geodetic.lon.deg, target.geodetic.lat.deg


# ===================================================================
# east_north_to_long_lat
# ===================================================================
class TestEastNorthToLongLat:
    """Tests for the ENU → geodetic coordinate conversion."""

    @pytest.mark.parametrize(
        "site_name, lon0, lat0, alt0",
        [
            ("LOFAR", LOFAR[0], LOFAR[1], LOFAR[2]),
            ("MeerKAT", MEERKAT[0], MEERKAT[1], MEERKAT[2]),
            ("ASKAP", ASKAP[0], ASKAP[1], ASKAP[2]),
            ("VLA", VLA[0], VLA[1], VLA[2]),
        ],
    )
    @pytest.mark.parametrize(
        "east, north, up",
        [
            (1000, 1000, 0),
            (-1000, -1000, 0),
            (5000, -2000, 0),
            (0, 3000, 0),
            (-4000, 0, 0),
            (1000, 1000, 50),
            (0, 0, 100),
        ],
        ids=[
            "NE_1km",
            "SW_1km",
            "E5km_S2km",
            "N_3km",
            "W_4km",
            "NE_1km_up50",
            "up_only_100",
        ],
    )
    def test_against_astropy(
        self,
        site_name: str,
        lon0: float,
        lat0: float,
        alt0: float,
        east: float,
        north: float,
        up: float,
    ) -> None:
        """ENU conversion must match astropy at multiple sites and offsets."""
        result_lon, result_lat = east_north_to_long_lat(
            east, north, lon0, lat0, alt=alt0, up=up
        )
        expected_lon, expected_lat = _astropy_enu_to_geodetic(
            east, north, lon0, lat0, alt0=alt0, up=up
        )
        assert abs(result_lon - expected_lon) < GEODETIC_ATOL_DEG, (
            f"{site_name}: lon mismatch {result_lon} vs {expected_lon}"
        )
        assert abs(result_lat - expected_lat) < GEODETIC_ATOL_DEG, (
            f"{site_name}: lat mismatch {result_lat} vs {expected_lat}"
        )

    def test_zero_offset_is_identity(self) -> None:
        """A zero ENU offset must return the reference point exactly."""
        lon0, lat0 = LOFAR[0], LOFAR[1]
        result_lon, result_lat = east_north_to_long_lat(0.0, 0.0, lon0, lat0)
        assert abs(result_lon - lon0) < GEODETIC_ATOL_DEG
        assert abs(result_lat - lat0) < GEODETIC_ATOL_DEG

    def test_east_only_changes_longitude(self) -> None:
        """A pure eastward displacement must not change latitude appreciably."""
        lon0, lat0 = MEERKAT[0], MEERKAT[1]
        result_lon, result_lat = east_north_to_long_lat(1000.0, 0.0, lon0, lat0)
        assert result_lon > lon0  # moved east → longitude increases
        assert abs(result_lat - lat0) < 1e-6  # latitude essentially unchanged

    def test_north_only_changes_latitude(self) -> None:
        """A pure northward displacement must not change longitude appreciably."""
        lon0, lat0 = MEERKAT[0], MEERKAT[1]
        result_lon, result_lat = east_north_to_long_lat(0.0, 1000.0, lon0, lat0)
        assert result_lat > lat0  # moved north → latitude increases
        assert abs(result_lon - lon0) < 1e-6  # longitude essentially unchanged

    def test_symmetry(self) -> None:
        """Opposite offsets must be approximately symmetric around the reference.

        On an oblate ellipsoid the symmetry is not exact because the curvature
        varies with latitude, so we allow a tolerance of ~1e-5 deg (~1 m).
        """
        lon0, lat0 = ASKAP[0], ASKAP[1]
        lon_p, lat_p = east_north_to_long_lat(2000, 3000, lon0, lat0)
        lon_m, lat_m = east_north_to_long_lat(-2000, -3000, lon0, lat0)
        # Midpoint should be close to the reference
        assert abs((lon_p + lon_m) / 2 - lon0) < 1e-5
        assert abs((lat_p + lat_m) / 2 - lat0) < 1e-5

    def test_altitude_affects_result(self) -> None:
        """A high-altitude reference must produce different lon/lat than sea level.

        For a site like the VLA at 2124 m, the ECEF origin shifts, which means
        the same ENU offset maps to slightly different geodetic coordinates.
        """
        lon0, lat0, alt0 = VLA
        east, north = 5000.0, 3000.0
        lon_sea, lat_sea = east_north_to_long_lat(east, north, lon0, lat0, alt=0.0)
        lon_alt, lat_alt = east_north_to_long_lat(east, north, lon0, lat0, alt=alt0)
        # The difference should be small but non-zero
        assert (lon_sea, lat_sea) != (lon_alt, lat_alt)

    def test_up_component_preserves_lonlat_at_zero_en(self) -> None:
        """A pure upward offset should barely change lon/lat."""
        lon0, lat0, alt0 = MEERKAT
        result_lon, result_lat = east_north_to_long_lat(
            0.0, 0.0, lon0, lat0, alt=alt0, up=500.0
        )
        assert abs(result_lon - lon0) < 1e-6
        assert abs(result_lat - lat0) < 1e-6

    def test_backward_compatible_defaults(self) -> None:
        """Calling without alt/up must produce the same result as the old API."""
        lon0, lat0 = LOFAR[0], LOFAR[1]
        lon_new, lat_new = east_north_to_long_lat(1000, 2000, lon0, lat0)
        lon_explicit, lat_explicit = east_north_to_long_lat(
            1000, 2000, lon0, lat0, alt=0.0, up=0.0
        )
        assert lon_new == lon_explicit
        assert lat_new == lat_explicit


# ===================================================================
# wgs84_to_cartesian
# ===================================================================
class TestWgs84ToCartesian:
    """Tests for the public WGS84 → ECEF conversion (supports arrays)."""

    @pytest.mark.parametrize("lon, lat, alt", SITES, ids=["LOFAR", "MeerKAT",
                                                          "ASKAP", "VLA"])
    def test_against_astropy(self, lon: float, lat: float, alt: float) -> None:
        """Scalar input must match astropy to sub-millimetre accuracy."""
        result: NDArray[np.float64] = wgs84_to_cartesian(lon, lat, alt)
        expected = _astropy_ecef(lon, lat, alt)
        np.testing.assert_allclose(result.flatten(), expected, atol=ECEF_ATOL_M)

    def test_output_shape_scalar(self) -> None:
        """Scalar inputs must produce a (1, 3) array."""
        result = wgs84_to_cartesian(0.0, 0.0, 0.0)
        assert result.shape == (1, 3) or result.shape == (3,)

    def test_vectorised_input(self) -> None:
        """Array inputs must be handled element-wise, matching astropy."""
        lons = np.array([s[0] for s in SITES])
        lats = np.array([s[1] for s in SITES])
        alts = np.array([s[2] for s in SITES])

        result: NDArray[np.float64] = wgs84_to_cartesian(lons, lats, alts)
        assert result.shape == (len(SITES), 3)

        for i, (lon, lat, alt) in enumerate(SITES):
            expected = _astropy_ecef(lon, lat, alt)
            np.testing.assert_allclose(result[i], expected, atol=ECEF_ATOL_M)

    def test_no_radius_parameter(self) -> None:
        """The old `radius` parameter has been removed; passing it must raise."""
        with pytest.raises(TypeError):
            wgs84_to_cartesian(0.0, 0.0, 0.0, radius=6_378_100)
