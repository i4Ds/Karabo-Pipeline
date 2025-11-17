from __future__ import annotations

from enum import Enum
from typing import TypeVar

from karabo.imaging.imager_base import DirtyImager, DirtyImagerConfig
from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig


class ImagingBackend(str, Enum):
    """Supported imaging backends."""

    RASCIL = "rascil"
    SDP = "sdp"


ConfigT = TypeVar("ConfigT", bound=DirtyImagerConfig)


def get_imager(backend: ImagingBackend, config: ConfigT) -> DirtyImager:
    """Instantiate an imager for the requested backend.

    Args:
        backend: Imaging backend to use.
        config: Configuration object for the imager. General parameters are
            coerced to backend-specific configs when possible.

    Returns:
        DirtyImager: Imager instance bound to the requested backend.

    Raises:
        NotImplementedError: If the backend is not yet implemented.
        ValueError: If the backend value is not recognised.
    """

    if backend is ImagingBackend.RASCIL:
        if isinstance(config, RascilDirtyImagerConfig):
            rascil_config = config
        else:
            rascil_config = RascilDirtyImagerConfig(
                imaging_npixel=config.imaging_npixel,
                imaging_cellsize=config.imaging_cellsize,
                combine_across_frequencies=config.combine_across_frequencies,
            )
        return RascilDirtyImager(rascil_config)

    if backend is ImagingBackend.SDP:
        raise NotImplementedError("SKA-SDP imaging backend not yet implemented.")

    raise ValueError(f"Unsupported imaging backend requested: {backend!r}")
