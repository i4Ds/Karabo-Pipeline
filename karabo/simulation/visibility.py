from __future__ import annotations

import os
import os.path
from typing import Callable, Dict, Final, List, Literal, Optional, Union, get_args

import numpy as np
import oskar

from karabo.util._types import DirPathType, FilePathType
from karabo.util.file_handler import FileHandler

# If you add a new format, a corresponding path validator function needs to be added
# to _VISIBILITY_FORMAT_VALIDATORS.
VisibilityFormat = Literal["MS", "OSKAR_VIS", "UVFITS"]
_VISIBILITY_FORMAT_VALIDATORS: Final[
    Dict[VisibilityFormat, Callable[[Union[DirPathType, FilePathType]], bool]]
] = {
    "MS": lambda path: str(path).lower().endswith(".ms"),
    "OSKAR_VIS": lambda path: str(path).lower().endswith(".vis"),
    "UVFITS": lambda path: (lower := str(path).lower()).endswith(".uvfits")
    or lower.endswith(".uvf"),
}
assert len(get_args(VisibilityFormat)) == len(_VISIBILITY_FORMAT_VALIDATORS)
assert all(f in _VISIBILITY_FORMAT_VALIDATORS for f in get_args(VisibilityFormat))


class VisibilityFormatUtil:
    @classmethod
    def is_valid_path_for_format(
        cls,
        path: Union[DirPathType, FilePathType],
        format: VisibilityFormat,
    ) -> bool:
        """Tests if a path is valid for a specific format

        Args:
            path: path to visibility
            format: visibility format

        Returns:
            True if valid, False otherwise
        """
        return _VISIBILITY_FORMAT_VALIDATORS[format](path)

    @classmethod
    def parse_visibility_format_from_path(
        cls,
        path: Union[DirPathType, FilePathType],
    ) -> Optional[VisibilityFormat]:
        for format in sorted(_VISIBILITY_FORMAT_VALIDATORS.keys()):
            if cls.is_valid_path_for_format(path, format):
                return format
        return None


class Visibility:
    """Class representing visibility data on the filesystem

    Args:
        path: Path to visibility data directory (for MS format) or file.
            Visibility format will be inferred from the path.
    """

    def __init__(self, path: Union[DirPathType, FilePathType]) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        self.path = path

        self.format: VisibilityFormat
        format = VisibilityFormatUtil.parse_visibility_format_from_path(self.path)
        if format is None:
            raise ValueError(
                f"Could not match {self.path} to one of the supported "
                f"visibility formats {get_args(VisibilityFormat)}"
            )
        self.format = format
        print(f"Matched path {self.path} to format {self.format}")

    # TODO Support MS, then adjust InterferometerSimulation.__run_simulation_long
    # to support both formats.
    @classmethod
    def combine_vis(
        cls,
        visibilities: List[Visibility],
        combined_ms_filepath: Optional[DirPathType] = None,
        group_by: str = "day",
    ) -> DirPathType:
        if not all(v.format == "OSKAR_VIS" for v in visibilities):
            raise NotImplementedError("Only OSKAR_VIS visibilities supported")

        print(f"Combining {len(visibilities)} visibilities...")
        if combined_ms_filepath is None:
            tmp_dir = FileHandler().get_tmp_dir(
                prefix="combine-vis-",
                purpose="combine-vis disk-cache.",
            )
            combined_ms_filepath = os.path.join(tmp_dir, "combined.MS")

        # Initialize lists to store data
        out_vis, uui, vvi, wwi, time_start, time_inc, time_ave = ([] for _ in range(7))

        # Loop over visibility files and read data
        for visibility in visibilities:
            (header, handle) = oskar.VisHeader.read(str(visibility.path))
            block = oskar.VisBlock.create_from_header(header)
            for k in range(header.num_blocks):
                block.read(header, handle, k)
            out_vis.append(block.cross_correlations())
            uui.append(block.baseline_uu_metres())
            vvi.append(block.baseline_vv_metres())
            wwi.append(block.baseline_ww_metres())
            time_inc.append(header.time_inc_sec)
            time_start.append(header.time_start_mjd_utc)
            time_ave.append(header.get_time_average_sec())

        # Combine visibility data
        ms = oskar.MeasurementSet.create(
            str(combined_ms_filepath),
            block.num_stations,
            block.num_channels,
            block.num_pols,
            header.freq_start_hz,
            header.freq_inc_hz,
        )
        deg2rad = np.pi / 180
        ms.set_phase_centre(
            header.phase_centre_ra_deg * deg2rad, header.phase_centre_dec_deg * deg2rad
        )

        # Write combined visibility data
        print(f"### Writing combined visibilities in {combined_ms_filepath} ...")

        num_files = len(visibilities)
        if group_by == "day":
            for j in range(num_files):
                num_times = out_vis[j].shape[0]
                for t in range(num_times):
                    time_stamp = time_inc[j] * time_start[j]
                    exposure_sec = time_ave[0]
                    start_row = t * block.num_baselines
                    ms.write_coords(
                        start_row,
                        block.num_baselines,
                        uui[j][t],
                        vvi[j][t],
                        wwi[j][t],
                        exposure_sec,
                        time_ave[j],
                        time_stamp,
                    )
                    ms.write_vis(
                        start_row,
                        0,
                        block.num_channels,
                        block.num_baselines,
                        out_vis[j][t],
                    )
        else:
            num_times = out_vis[0].shape[0] * num_files
            ushape = np.array(uui).shape
            outshape = np.array(out_vis).shape
            uuf = np.array(uui).reshape(ushape[0] * ushape[1], ushape[2])
            vvf = np.array(vvi).reshape(ushape[0] * ushape[1], ushape[2])
            wwf = np.array(wwi).reshape(ushape[0] * ushape[1], ushape[2])
            out_vis_reshaped = np.array(out_vis).reshape(
                outshape[0] * outshape[1], outshape[2], outshape[3], outshape[4]
            )
            for t in range(num_times):
                time_stamp = time_start[0] + t * time_inc[0] / 86400.0
                exposure_sec = time_ave[0]
                interval_sec = time_ave[0]
                start_row = t * block.num_baselines

                ms.write_coords(
                    start_row,
                    block.num_baselines,
                    np.mean(uuf, axis=0),
                    np.mean(vvf, axis=0),
                    np.mean(wwf, axis=0),
                    exposure_sec,
                    interval_sec,
                    time_stamp,
                )
                ms.write_vis(
                    start_row,
                    0,
                    block.num_channels,
                    block.num_baselines,
                    out_vis_reshaped[t],
                )

        return combined_ms_filepath
