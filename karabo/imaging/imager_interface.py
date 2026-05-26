from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from karabo.imaging.image import Image
from karabo.simulation.visibility import Visibility


@dataclass(frozen=True)
class ImageSpec:
    """Minimal imaging parameters shared across backends."""

    npix: int
    cellsize_arcsec: float
    phase_centre_deg: Tuple[float, float]
    polarisation: str = "I"
    nchan: int = 1

    @property
    def cellsize_radians(self) -> float:
        """Return the pixel scale in radians."""
        return self.cellsize_arcsec * (math.pi / (180.0 * 3600.0))


class Imager(ABC):
    """Backend-agnostic imaging interface."""

    @abstractmethod
    def invert(self, vis: Visibility, image_spec: ImageSpec) -> tuple[Image, Image]:
        """Grid visibilities into dirty image and PSF images."""

    @abstractmethod
    def restore(self, model_or_dirty: Image, psf: Image) -> Image:
        """Return a restored image using the provided PSF."""
