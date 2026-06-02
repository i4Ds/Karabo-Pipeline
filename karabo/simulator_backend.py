import enum


class SimulatorBackend(enum.Enum):
    OSKAR = "OSKAR"
    RASCIL = "RASCIL"
    SDP = "ska-sdp"
