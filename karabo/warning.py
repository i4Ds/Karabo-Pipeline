from __future__ import annotations


class KaraboWarning(Warning):
    """
    Base Warning thrown by the Karabo Pipeline
    """


class InterferometerSimulationWarning(KaraboWarning):
    """
    Warning thrown by the Interferometer Simulation
    """


_DEV_ERROR_MSG = (
    "This is an implementation error from Karabo. "
    + "Please open an issue at https://github.com/i4Ds/Karabo-Pipeline"
)
