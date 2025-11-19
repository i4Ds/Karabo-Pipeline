from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from karabo.imaging.backends.rascil_backend import RascilBackendImager
from karabo.imaging.backends.sdp_backend import SdpImager
from karabo.imaging.imager_interface import Imager
from karabo.util.helpers import Environment


class ImagingBackend(str, Enum):
    """Supported imaging backends."""

    RASCIL = "rascil"
    SDP = "sdp"


def parse_imaging_backend(
    backend: Optional[Union[str, ImagingBackend]] = None,
) -> ImagingBackend:
    if backend is None:
        backend = Environment.get("IMAGING_BACKEND", str, ImagingBackend.RASCIL.value)
    if isinstance(backend, ImagingBackend):
        return backend
    backend_lower = backend.lower()
    try:
        return ImagingBackend(backend_lower)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported imaging backend '{backend}'. Expected one of: "
            f"{', '.join(b.value for b in ImagingBackend)}"
        ) from exc


def get_imager(backend: Optional[Union[str, ImagingBackend]] = None) -> Imager:
    resolved = parse_imaging_backend(backend)
    if resolved is ImagingBackend.RASCIL:
        return RascilBackendImager()
    if resolved is ImagingBackend.SDP:
        return SdpImager()
    raise ValueError(f"Unsupported imaging backend requested: {resolved!r}")
