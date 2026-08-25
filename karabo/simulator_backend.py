import enum


class SimulatorBackend(enum.Enum):
    """Supported interferometer simulation backends."""

    OSKAR = "OSKAR"
    SDP = "ska-sdp"

    @classmethod
    def _missing_(cls, value: object) -> None:
        if isinstance(value, str) and value.lower() == "rascil":
            raise ValueError(
                "The RASCIL simulation backend has been removed. "
                "Use SimulatorBackend.SDP instead."
            )
        return None
