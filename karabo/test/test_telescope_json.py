import filecmp
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest
from casacore.tables import table

from karabo.error import KaraboError
from karabo.simulation import telescope_json
from karabo.simulation.interferometer import InterferometerSimulation
from karabo.simulation.observation import Observation
from karabo.simulation.sky_model import SkyModel
from karabo.simulation.telescope import Telescope
from karabo.simulation.telescope_versions import ALMAVersions, SKAMidAAStarVersions
from karabo.simulator_backend import SimulatorBackend
from karabo.util.data_util import get_module_absolute_path

SKA_MID_AASTAR_JSON = os.path.join(
    get_module_absolute_path(),
    "data",
    "SKA-MID-AAstar.ska-ost-array-config-3.2.0.json",
)

SEFD = {
    "Band 1": [[0.35e9, 1671.3], [0.7e9, 550.9]],
    "Band 2": [[0.95e9, 326.8], [1.76e9, 281.3]],
}


def _ska_mid_aastar_tm() -> Telescope:
    return Telescope.constructor(
        "SKA-MID-AAstar", SKAMidAAStarVersions.SKA_OST_ARRAY_CONFIG_2_3_1
    )


def _telescope_with_metadata() -> Telescope:
    tel = Telescope(21.44, -30.71, 1050.0)
    tel.name = "test-telescope"
    tel.sefd_reference = {"source": "test", "elevation_deg": 45.0}
    tel.add_station(0.0, 0.0, 0.0, name="A0", dish_diameter_m=15.0, sefd=SEFD)
    tel.add_station(100.0, 50.0, 1.0, 0.1, 0.2, 0.3, name="A1")
    tel.add_station(-30.0, 20.0)
    tel.add_antenna_to_station(0, 0.1, 0.1)
    tel.add_antenna_to_station(1, 0.1, 0.1)
    tel.add_antenna_to_station(2, 1.0, 2.0, 0.5, 0.01, 0.02, 0.03)
    tel.add_antenna_to_station(2, -1.0, -2.0)
    return tel


def _geometry(tel: Telescope) -> list:
    def row(c):
        return [c.x, c.y, c.z, c.x_error, c.y_error, c.z_error]

    return [(row(s.position), [row(a) for a in s.antennas]) for s in tel.stations]


def _assert_same_tm(dir_a: str, dir_b: str) -> None:
    cmp = filecmp.dircmp(dir_a, dir_b)
    assert sorted(cmp.left_list) == sorted(cmp.right_list)
    _, mismatch, errors = filecmp.cmpfiles(
        dir_a, dir_b, ["position.txt", "layout.txt"], shallow=False
    )
    assert not mismatch and not errors
    for station in cmp.common_dirs:
        _, mismatch, errors = filecmp.cmpfiles(
            os.path.join(dir_a, station),
            os.path.join(dir_b, station),
            ["layout.txt"],
            shallow=False,
        )
        assert not mismatch and not errors, station


def test_station_directories_are_read_in_order(tmp_path):
    tel = Telescope(0.0, 0.0)
    for i in range(12):
        tel.add_station(10.0 * i, 0.0)
        for j in range(i + 1):
            tel.add_antenna_to_station(i, float(j), 0.0)
    tm_path = tmp_path / "heterogeneous.tm"
    tel.write_to_disk(tm_path)

    read = Telescope.read_OSKAR_tm_file(tm_path)
    for i, station in enumerate(read.stations):
        assert station.position.x == 10.0 * i
        assert len(station.antennas) == i + 1


def test_json_round_trip(tmp_path):
    tel = _telescope_with_metadata()
    path = tmp_path / "tel.json"
    tel.to_json(path)
    read = Telescope.from_json(path)

    assert read.name == "test-telescope"
    assert read.backend is SimulatorBackend.OSKAR
    assert read.centre_altitude == 1050.0
    assert read.sefd_reference == tel.sefd_reference
    assert _geometry(read) == _geometry(tel)
    assert [s.name for s in read.stations] == ["A0", "A1", None]
    assert [s.dish_diameter_m for s in read.stations] == [15.0, None, None]
    assert read.stations[1].sefd is None
    for band, samples in SEFD.items():
        np.testing.assert_array_equal(read.stations[0].sefd[band], samples)
    assert telescope_json.to_dict(read) == telescope_json.to_dict(tel)


def test_json_shares_identical_layouts_and_sefds():
    tel = Telescope(0.0, 0.0)
    for i in range(4):
        tel.add_station(float(i), 0.0, sefd=SEFD if i < 3 else None)
        tel.add_antenna_to_station(i, 0.0, 0.0)
    data = telescope_json.to_dict(tel)
    assert data["element_layouts"] == {"layout0": [[0.0, 0.0, 0.0]]}
    assert list(data["sefd_curves"]) == ["sefd0"]
    assert [s.get("sefd") for s in data["stations"]] == ["sefd0"] * 3 + [None]


def test_json_station_without_elements_has_single_element(tmp_path):
    data = {
        "format": "karabo-telescope",
        "version": 1,
        "centre": {"longitude_deg": 0.0, "latitude_deg": 0.0},
        "stations": [{"position_m": [1.0, 2.0, 3.0]}],
    }
    path = tmp_path / "tel.json"
    path.write_text(json.dumps(data))
    tel = Telescope.from_json(path)
    assert tel.centre_altitude == 0.0
    assert tel.name == "telescope"
    assert _geometry(tel) == [([1.0, 2.0, 3.0, 0, 0, 0], [[0, 0, 0, 0, 0, 0]])]


@pytest.mark.parametrize(
    "name, version",
    [
        ("EXAMPLE", None),
        ("MeerKAT", None),
        ("SKA1LOW", None),
        ("ALMA", ALMAVersions.CYCLE_1_1),
        ("SKA-MID-AAstar", SKAMidAAStarVersions.SKA_OST_ARRAY_CONFIG_2_3_1),
    ],
)
def test_legacy_tm_to_json_keeps_geometry(tmp_path, name, version):
    legacy = Telescope.constructor(name, version)
    path = tmp_path / "tel.json"
    legacy.to_json(path)
    from_json = Telescope.from_json(path)

    assert from_json.name == legacy.name
    assert _geometry(from_json) == _geometry(legacy)
    assert all(s.name is None and s.sefd is None for s in from_json.stations)
    legacy_tm = str(tmp_path / "legacy.tm")
    legacy.write_to_disk(legacy_tm)
    _assert_same_tm(legacy_tm, str(from_json.path))


def test_ska1low_json_uses_one_shared_layout(tmp_path):
    tel = Telescope.constructor("SKA1LOW")
    path = tmp_path / "ska1low.json"
    tel.to_json(path)
    data = json.loads(path.read_text())
    assert len(data["element_layouts"]) == 1
    assert {s["elements"] for s in data["stations"]} == {"layout0"}
    assert os.path.getsize(path) < 1_000_000


def test_generated_tm_contains_only_geometry_files(tmp_path):
    path = tmp_path / "tel.json"
    _telescope_with_metadata().to_json(path)
    tel = Telescope.from_json(path)
    assert sorted(os.listdir(tel.path)) == [
        "layout.txt",
        "position.txt",
        "station000",
        "station001",
        "station002",
    ]
    for station in ["station000", "station001", "station002"]:
        assert os.listdir(os.path.join(tel.path, station)) == ["layout.txt"]


def test_metadata_does_not_change_write_to_disk(tmp_path):
    with_metadata = _telescope_with_metadata()
    without_metadata = _telescope_with_metadata()
    for station in without_metadata.stations:
        station.name = None
        station.dish_diameter_m = None
        station.sefd = None
    with_metadata.write_to_disk(tmp_path / "with.tm")
    without_metadata.write_to_disk(tmp_path / "without.tm")
    _assert_same_tm(str(tmp_path / "with.tm"), str(tmp_path / "without.tm"))


def test_get_OSKAR_telescope_from_json():
    tel = Telescope.from_json(SKA_MID_AASTAR_JSON)
    assert tel.get_OSKAR_telescope().get_num_stations() == 144
    assert tel.stations[0].name == "M000"


def test_bundled_ska_mid_aastar_json():
    tel = Telescope.from_json(SKA_MID_AASTAR_JSON)
    assert tel.name == "SKA-MID-AAstar"
    assert len(tel.stations) == 144
    assert _geometry(tel) == _geometry(_ska_mid_aastar_tm())

    ska = [s for s in tel.stations if s.name.startswith("SKA")]
    meerkat = [s for s in tel.stations if s.name.startswith("M")]
    assert len(ska) == 80 and len(meerkat) == 64
    assert {s.dish_diameter_m for s in ska} == {15.0}
    assert {s.dish_diameter_m for s in meerkat} == {13.5}
    assert set(ska[0].sefd) == {"Band 1", "Band 2", "Band 5a", "Band 5b"}
    assert set(meerkat[0].sefd) == {"Band 1", "Band 2"}
    assert all(s.sefd.keys() == ska[0].sefd.keys() for s in ska)

    # SKA dishes are more sensitive than MeerKAT dishes in the shared L-band
    ska_band2 = ska[0].sefd["Band 2"]
    meerkat_band2 = meerkat[0].sefd["Band 2"]
    assert ska_band2[0, 0] == 0.95e9
    assert 250 < ska_band2[:, 1].min() and ska_band2[:, 1].max() < 350
    assert np.all(
        np.interp(meerkat_band2[:, 0], ska_band2[:, 0], ska_band2[:, 1])
        < meerkat_band2[:, 1]
    )
    assert tel.sefd_reference["source"] == "ska-ost-senscalc 12.0.1"


def test_sdp_telescope_has_names_and_diameters():
    tel = Telescope.constructor("MID", backend=SimulatorBackend.SDP)
    config = tel.SDP_configuration
    assert [s.name for s in tel.stations] == list(config.names.data)
    assert [s.dish_diameter_m for s in tel.stations] == list(config.diameter.data)
    assert all(s.sefd is None for s in tel.stations)


def test_oskar_simulation_from_json_matches_tm():
    observation = Observation(
        phase_centre_ra_deg=20,
        phase_centre_dec_deg=-30,
        start_date_and_time=datetime(2000, 3, 20, 12, 6, 39),
        length=timedelta(minutes=10),
        number_of_time_steps=2,
        start_frequency_hz=1.4e9,
        frequency_increment_hz=1e7,
        number_of_channels=1,
    )
    simulation = InterferometerSimulation(
        channel_bandwidth_hz=1e7,
        time_average_sec=8,
        ignore_w_components=True,
        use_gpus=False,
        use_dask=False,
    )
    data = []
    for tel in [_ska_mid_aastar_tm(), Telescope.from_json(SKA_MID_AASTAR_JSON)]:
        vis = simulation.run_simulation(tel, SkyModel.sky_test(), observation)
        with table(str(vis.path), ack=False) as ms:
            data.append(ms.getcol("DATA"))
    assert data[0].shape == (2 * 144 * 143 // 2, 1, 4)
    assert np.any(data[0] != 0)
    np.testing.assert_array_equal(data[0], data[1])


@pytest.mark.parametrize(
    "change, match",
    [
        ({"format": "oskar"}, "format"),
        ({"version": 2}, "not supported"),
        ({"version": "1"}, "version"),
        ({"version": True}, "version"),
        ({"centre": {"latitude_deg": 0.0}}, "centre"),
        ({"stations": []}, "stations"),
        ({"stations": [{"position_m": [1.0, 2.0]}]}, "station 0"),
        ({"stations": [{"position_m": [1.0, 2.0, "a"]}]}, "station 0"),
        ({"stations": [{"position_m": [1.0, 2.0, float("nan")]}]}, "station 0"),
        ({"centre": {"longitude_deg": float("nan"), "latitude_deg": 0.0}}, "centre"),
        (
            {
                "centre": {
                    "longitude_deg": 0.0,
                    "latitude_deg": 0.0,
                    "altitude_m": 1e999,
                }
            },
            "centre",
        ),
        ({"stations": [{"position_m": "123"}]}, "station 0"),
        ({"stations": [{"position_m": [0, 0, 0], "elements": ["123"]}]}, "station 0"),
        ({"stations": [{"position_m": [0, 0, 0], "elements": []}]}, "station 0"),
        ({"stations": [{"position_m": [0, 0, 0], "elements": 5}]}, "station 0"),
        (
            {
                "element_layouts": {"empty": []},
                "stations": [{"position_m": [0, 0, 0], "elements": "empty"}],
            },
            "station 0",
        ),
        ({"stations": [{"name": "x"}]}, "station 0"),
        ({"stations": ["x"]}, "station 0"),
        (
            {"stations": [{"position_m": [0, 0, 0], "elements": "missing"}]},
            "station 0",
        ),
        (
            {"stations": [{"position_m": [0, 0, 0], "elements": [[0, 0]]}]},
            "station 0",
        ),
        (
            {"stations": [{"position_m": [0, 0, 0], "dish_diameter_m": -1}]},
            "station 0",
        ),
        (
            {
                "stations": [
                    {"position_m": [0, 0, 0], "sefd": {"B": [[2e9, 1], [1e9, 1]]}}
                ]
            },
            "station 0",
        ),
        (
            {"stations": [{"position_m": [0, 0, 0], "sefd": {"B": [[1e9, -1]]}}]},
            "station 0",
        ),
        (
            {"stations": [{"position_m": [0, 0, 0], "sefd": {"B": [1e9, 1]}}]},
            "station 0",
        ),
    ],
)
def test_invalid_json_raises(tmp_path, change, match):
    data = {
        "format": "karabo-telescope",
        "version": 1,
        "centre": {"longitude_deg": 0.0, "latitude_deg": 0.0},
        "stations": [{"position_m": [0.0, 0.0, 0.0]}],
    }
    data.update(change)
    path = tmp_path / "tel.json"
    path.write_text(json.dumps(data))
    with pytest.raises(KaraboError, match=match):
        Telescope.from_json(path)


@pytest.mark.parametrize("content", ["{", "[]"])
def test_malformed_json_raises(tmp_path, content):
    path = tmp_path / "tel.json"
    path.write_text(content)
    with pytest.raises(KaraboError):
        Telescope.from_json(path)


def test_unknown_keys_are_ignored(tmp_path):
    data = {
        "format": "karabo-telescope",
        "version": 1,
        "future_key": {"a": 1},
        "centre": {"longitude_deg": 0.0, "latitude_deg": 0.0},
        "stations": [{"position_m": [0.0, 0.0, 0.0], "future_station_key": 1}],
    }
    path = tmp_path / "tel.json"
    path.write_text(json.dumps(data))
    assert len(Telescope.from_json(path).stations) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dish_diameter_m": 0.0},
        {"dish_diameter_m": float("inf")},
        {"sefd": {"Band 1": [[1e9, 100.0], [1e9, 90.0]]}},
        {"sefd": {"Band 1": [[1e9, 0.0]]}},
        {"sefd": {"Band 1": [[1e9, np.nan]]}},
        {"sefd": {"Band 1": np.zeros((0, 2))}},
        {"sefd": {"Band 1": [[1e9, 100.0, 1.0]]}},
    ],
)
def test_invalid_station_metadata_raises(kwargs):
    tel = Telescope(0.0, 0.0)
    with pytest.raises(ValueError):
        tel.add_station(0.0, 0.0, **kwargs)


def test_sefd_arrays_are_read_only():
    tel = _telescope_with_metadata()
    with pytest.raises(ValueError, match="read-only"):
        tel.stations[0].sefd["Band 1"][0, 1] = -5.0
    tel.stations[0].sefd = {"Band 1": [[1e9, 50.0]]}
    np.testing.assert_array_equal(tel.stations[0].sefd["Band 1"], [[1e9, 50.0]])


def test_to_json_revalidates_sefd(tmp_path):
    tel = _telescope_with_metadata()
    tel.stations[0].sefd["Band 5a"] = [[1e9, -5.0]]
    with pytest.raises(ValueError, match="Band 5a"):
        tel.to_json(tmp_path / "tel.json")
    assert not (tmp_path / "tel.json").exists()


def test_to_json_rejects_invalid_geometry(tmp_path):
    tel = _telescope_with_metadata()
    tel.add_station(1.0, 1.0)
    with pytest.raises(KaraboError, match="Station 3 has no elements"):
        tel.to_json(tmp_path / "tel.json")

    tel = _telescope_with_metadata()
    tel.stations[1].position.x = float("nan")
    with pytest.raises(ValueError):
        tel.to_json(tmp_path / "tel.json")
