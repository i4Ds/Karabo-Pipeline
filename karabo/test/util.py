from karabo.imaging.imager_base import DirtyImager, DirtyImagerConfig
from karabo.imaging.imager_oskar import OskarDirtyImager, OskarDirtyImagerConfig
from karabo.imaging.imager_rascil import RascilDirtyImager, RascilDirtyImagerConfig
from karabo.imaging.imager_wsclean import WscleanDirtyImager
from karabo.simulation.visibility import Visibility


def get_compatible_dirty_imager(
    visibility: Visibility,
    config: DirtyImagerConfig,
) -> DirtyImager:
    """Automatically choose a suitable dirty imager based on a visibility object.

    Temporary function until we have a general visibility object
    and functions to convert general objects to implementation-specific
    objects on demand.

    Args:
        visibility: Visibility object
        config: Config to initialize dirty imager
            object with.

    Returns:
        DirtyImager: The created dirty imager object
    """
    dirty_imager: DirtyImager
    if visibility.format == "OSKAR_VIS":
        dirty_imager = OskarDirtyImager(
            OskarDirtyImagerConfig(
                imaging_npixel=config.imaging_npixel,
                imaging_cellsize=config.imaging_cellsize,
                combine_across_frequencies=config.combine_across_frequencies,
            )
        )
    else:
        if config.combine_across_frequencies is False:
            dirty_imager = RascilDirtyImager(
                RascilDirtyImagerConfig(
                    imaging_npixel=config.imaging_npixel,
                    imaging_cellsize=config.imaging_cellsize,
                    combine_across_frequencies=config.combine_across_frequencies,
                )
            )
        else:
            dirty_imager = WscleanDirtyImager(config)

    return dirty_imager
