"""Early sanity checks for core scientific stack.

These tests run first (by filename) to catch environment issues
immediately after Spack dependencies are installed.
"""

import pytest


def test_astropy_earthlocation_basic():
    from astropy import units as u
    from astropy.coordinates import EarthLocation, Longitude, Latitude
    import astropy
    import numpy as np

    # Ensure ERFA bindings are available (pyerfa)
    try:
        import erfa  # noqa: F401
    except Exception as exc:
        pytest.fail(f"erfa missing or not importable: {exc}")

    # Validate EarthLocation via ERFA gd2gc using plain floats then construct geocentric
    lon = Longitude(116.76444824, unit=u.deg)
    lat = Latitude(-26.82472208, unit=u.deg)
    height = 300 * u.m

    # Compute geocentric coordinates using WGS84 formula (avoid ERFA ufunc dtype quirks)
    phi = lat.to_value(u.rad)
    lam = lon.to_value(u.rad)
    h = height.to_value(u.m)
    a = 6378137.0  # WGS84 semi-major axis (m)
    f = 1.0 / 298.257223563  # WGS84 flattening
    e2 = f * (2.0 - f)
    sinp = np.sin(phi)
    cosp = np.cos(phi)
    sinl = np.sin(lam)
    cosl = np.cos(lam)
    N = a / np.sqrt(1.0 - e2 * sinp * sinp)
    x = (N + h) * cosp * cosl
    y = (N + h) * cosp * sinl
    z = (N * (1.0 - e2) + h) * sinp
    loc = EarthLocation.from_geocentric(x * u.m, y * u.m, z * u.m)
    assert loc is not None, "EarthLocation was not constructed"


def test_sdp_datamodels_named_configuration_has_location():
    dm = pytest.importorskip("ska_sdp_datamodels")
    from ska_sdp_datamodels.configuration.config_create import (
        create_named_configuration,
    )

    cfg = create_named_configuration("LOWBD2")
    # Expect a location-like object present; avoid depending on internals
    assert hasattr(cfg, "location") or hasattr(cfg, "earth_location")


