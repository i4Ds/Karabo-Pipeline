class KaraboError(Exception):
    """
    Base Exception thrown by the Karabo Pipeline
    """


class KaraboDaskError(KaraboError):
    ...


class KaraboInterferometerSimulationError(KaraboError):
    ...


class KaraboPinocchioError(KaraboError):
    ...


class KaraboSourceDetectionEvaluationError(KaraboError):
    ...


class NodeTermination(KaraboError):
    ...
