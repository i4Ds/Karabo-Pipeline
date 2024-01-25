"""Base signal class."""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from karabo.simulation.signal.typing import BaseImage

T = TypeVar("T", bound=BaseImage)


# pylint: disable=too-few-public-methods
class BaseSignal(Generic[T], ABC):
    """Base signal class."""

    @abstractmethod
    def simulate(self) -> list[T]:
        """Simulate a signal to get a 2D/3D image output."""
