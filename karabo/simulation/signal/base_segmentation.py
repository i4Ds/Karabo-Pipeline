"""Base segmentation class."""
from abc import ABC, abstractmethod

from karabo.simulation.signal.typing import Image3D, SegmentationOutput


# pylint: disable=too-few-public-methods
class BaseSegmentation(ABC):
    """Base segmentation class."""

    @abstractmethod
    def segment(self, image: Image3D) -> SegmentationOutput:
        """Segments on a 2D or 3D image."""
