class KaraboError(Exception):
    """
    Base Exception thrown by the Karabo Pipeline
    """


class KaraboDaskError(KaraboError):
    ...


class KaraboInterferometerSimulationError(KaraboError):
    ...


class NodeTermination(KaraboError):
    ...
