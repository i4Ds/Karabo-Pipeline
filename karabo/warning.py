from __future__ import annotations

from warnings import warn


class KaraboWarning(Warning):
    """
    Base Warning thrown by the Karabo Pipeline
    """


class InterferometerSimulationWarning(KaraboWarning):
    """
    Warning thrown by the Interferometer Simulation
    """


class RascilDeprecationWarning(KaraboWarning):
    """
    Warning thrown when users select deprecated RASCIL backends.
    """


RASCIL_DEPRECATION_MESSAGE = (
    "RASCIL support is deprecated and will be removed in a future release. "
    "Please use SDP instead."
)


def warn_rascil_deprecated(*, stacklevel: int = 2) -> None:
    """Warn that the selected RASCIL backend is deprecated."""
    warn(RASCIL_DEPRECATION_MESSAGE, RascilDeprecationWarning, stacklevel=stacklevel)


_DEV_ERROR_MSG = (
    "This is an implementation error from Karabo. "
    + "Please open an issue at https://github.com/i4Ds/Karabo-Pipeline"
)
