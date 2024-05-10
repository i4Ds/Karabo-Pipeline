from karabo.imaging.imager_base import DirtyImagerConfig
from karabo.imaging.imager_oskar import OskarDirtyImager, OskarDirtyImagerConfig


def test_constructor_from_dirty_imager_config():
    imaging_npixel = 1024
    imaging_cellsize = 0.1
    combine_across_frequencies = False

    dirty_imager = OskarDirtyImager(
        DirtyImagerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            combine_across_frequencies=combine_across_frequencies,
        )
    )
    assert isinstance(dirty_imager.config, OskarDirtyImagerConfig)
    assert dirty_imager.config.imaging_npixel == imaging_npixel
    assert dirty_imager.config.imaging_cellsize == imaging_cellsize
    assert dirty_imager.config.combine_across_frequencies == combine_across_frequencies

    dirty_imager = OskarDirtyImager(
        OskarDirtyImagerConfig(
            imaging_npixel=imaging_npixel,
            imaging_cellsize=imaging_cellsize,
            combine_across_frequencies=combine_across_frequencies,
            imaging_phasecentre=None,
        )
    )
    assert isinstance(dirty_imager.config, OskarDirtyImagerConfig)
    assert dirty_imager.config.imaging_npixel == imaging_npixel
    assert dirty_imager.config.imaging_cellsize == imaging_cellsize
    assert dirty_imager.config.combine_across_frequencies == combine_across_frequencies
