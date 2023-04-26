class KaraboError(Exception):
    """
    Base Exception thrown by the Karabo Pipeline
    """


class KaraboDaskError(KaraboError):
    ...


class NodeTermination(KaraboError):
    ...
