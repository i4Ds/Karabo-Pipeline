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


class DirectWscleanUsageWarning(KaraboWarning):
    """
    Warning thrown when users use WSClean outside the imager backend interface.
    """


RASCIL_DEPRECATION_MESSAGE = (
    "RASCIL support is deprecated and will be removed in a future release. "
    "Please use SDP instead."
)

DIRECT_WSCLEAN_USAGE_MESSAGE = (
    "Direct WSClean imaging classes are deprecated. "
    "Please use get_imager(ImagingBackend.WSCLEAN) instead."
)


def warn_rascil_deprecated(*, stacklevel: int = 2) -> None:
    """Warn that the selected RASCIL backend is deprecated."""
    warn(RASCIL_DEPRECATION_MESSAGE, RascilDeprecationWarning, stacklevel=stacklevel)


def warn_direct_wsclean_use(*, stacklevel: int = 2) -> None:
    """Warn that direct WSClean use should go through the imager backend."""
    warn(
        DIRECT_WSCLEAN_USAGE_MESSAGE,
        DirectWscleanUsageWarning,
        stacklevel=stacklevel,
    )


_DEV_ERROR_MSG = (
    "This is an implementation error from Karabo. "
    + "Please open an issue at https://github.com/i4Ds/Karabo-Pipeline"
)
