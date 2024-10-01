import os

import pytest

from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_oskar import OskarDirtyImager, OskarDirtyImagerConfig
from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.imaging.imager_wsclean import WscleanDirtyImager
from karabo.simulation.sample_simulation import run_sample_simulation
from karabo.simulation.visibility import VisibilityFormat
from karabo.simulator_backend import SimulatorBackend

IMAGING_NPIXEL = 2048
IMAGING_CELLSIZE = 3.878509448876288e-05


@pytest.mark.parametrize(
    "simulator_backend,visibility_format",
    [
        (SimulatorBackend.OSKAR, "MS"),
        (SimulatorBackend.OSKAR, "OSKAR_VIS"),
        # (SimulatorBackend.RASCIL, "MS"),
    ],
)
def test_oskar_imager(
    simulator_backend: SimulatorBackend, visibility_format: VisibilityFormat
) -> None:
    visibility, _, _, _, _, _ = run_sample_simulation(
        simulator_backend=simulator_backend,
        visibility_format=visibility_format,
    )
    assert os.path.exists(visibility.path)
    dirty_image = OskarDirtyImager(
        OskarDirtyImagerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,
        )
    ).create_dirty_image(visibility)
    assert os.path.isfile(dirty_image.path)


@pytest.mark.parametrize(
    "simulator_backend,visibility_format",
    [
        (SimulatorBackend.OSKAR, "MS"),
        # (SimulatorBackend.RASCIL, "MS"),
    ],
)
def test_rascil_imager(
    simulator_backend: SimulatorBackend, visibility_format: VisibilityFormat
) -> None:
    visibility, _, _, _, _, _ = run_sample_simulation(
        simulator_backend=simulator_backend,
        visibility_format=visibility_format,
    )
    assert os.path.exists(visibility.path)
    dirty_image = RascilDirtyImager(
        RascilDirtyImagerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,
        )
    ).create_dirty_image(visibility)
    assert os.path.isfile(dirty_image.path)


@pytest.mark.parametrize(
    "simulator_backend,visibility_format",
    [
        (SimulatorBackend.OSKAR, "MS"),
        # (SimulatorBackend.RASCIL, "MS"),
    ],
)
def test_wsclean_imager(
    simulator_backend: SimulatorBackend, visibility_format: VisibilityFormat
) -> None:
    visibility, _, _, _, _, _ = run_sample_simulation(
        simulator_backend=simulator_backend,
        visibility_format=visibility_format,
    )
    assert os.path.exists(visibility.path)
    dirty_image = WscleanDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=IMAGING_NPIXEL,
            imaging_cellsize=IMAGING_CELLSIZE,
        )
    ).create_dirty_image(visibility)
    assert os.path.isfile(dirty_image.path)
