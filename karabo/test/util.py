import math

from karabo.imaging.backends.sdp_backend import SdpImager, SdpImagerConfig
from karabo.imaging.image import Image
from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_factory import ImagingBackend, get_imager
from karabo.imaging.imager_interface import ImageSpec
from karabo.imaging.imager_oskar import OskarDirtyImager, OskarDirtyImagerConfig
from karabo.simulation.visibility import Visibility


def create_compatible_dirty_image(
    visibility: Visibility,
    config: DirtyImagerConfig,
) -> Image:
    """Create a dirty image for tests that exercise both visibility formats.

    OSKAR_VIS still requires the OSKAR-native imager. Measurement Sets use the
    backend-neutral imaging API, selecting SDP when channel separation is required
    and WSClean otherwise.
    """
    if visibility.format == "OSKAR_VIS":
        return OskarDirtyImager(
            OskarDirtyImagerConfig(
                imaging_npixel=config.imaging_npixel,
                imaging_cellsize=config.imaging_cellsize,
                combine_across_frequencies=config.combine_across_frequencies,
            )
        ).create_dirty_image(visibility)

    if config.combine_across_frequencies:
        imager = get_imager(ImagingBackend.WSCLEAN)
    else:
        imager = SdpImager(SdpImagerConfig(combine_across_frequencies=False))
    dirty, _ = imager.invert(
        visibility,
        ImageSpec(
            npix=config.imaging_npixel,
            cellsize_arcsec=math.degrees(config.imaging_cellsize) * 3600.0,
            phase_centre_deg=(0.0, 0.0),
        ),
    )
    return dirty
