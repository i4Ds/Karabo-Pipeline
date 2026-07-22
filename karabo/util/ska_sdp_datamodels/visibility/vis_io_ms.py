# type: ignore
# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
# pylint: disable=too-many-nested-blocks,too-many-branches
# pylint: disable=invalid-name, duplicate-code
"""
Source: https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels
Copyright: SKAO
License: Apache License 2.0

Base functions to create and export Visibility
from/into Measurement Set files.
They take definitions of columns from msv2.py
and interact with Casacore.
"""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.units import Quantity
from ska_sdp_datamodels.configuration.config_model import Configuration
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
    ReceptorFrame,
)
from ska_sdp_datamodels.visibility.vis_model import Visibility
from ska_sdp_datamodels.visibility.vis_utils import generate_baselines


def _polarisation_frame_from_corr_type(corr_type):
    corr_type = np.sort(corr_type)
    frames = {
        (1, 2, 3, 4): "stokesIQUV",
        (1, 2): "stokesIQ",
        (1, 4): "stokesIV",
        (5, 6, 7, 8): "circular",
        (5, 8): "circularnp",
        (9, 10, 11, 12): "linear",
        (9, 12): "linearnp",
        (1,): "stokesI",
        (9,): "stokesI",
    }
    try:
        return PolarisationFrame(frames[tuple(corr_type.tolist())])
    except KeyError as exc:
        raise KeyError(f"Polarisation not understood: {corr_type}") from exc


def import_visibility_from_ms(msname, ack=False, datacolumn="DATA"):
    """Read Measurement Set fields and spectral windows as SDP visibilities.

    The pinned ska-sdp-datamodels release does not yet provide its later MS reader.
    This reader covers the single-field/spectral-window inputs supported by the SDP
    imager while returning a list so unsupported multi-block inputs can be rejected
    explicitly by the caller.
    """
    try:
        from casacore.tables import table
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("python-casacore is required for MS input") from exc

    root = table(msname, ack=ack)
    visibilities = []
    try:
        fields = np.unique(root.getcol("FIELD_ID"))
        data_descriptions = np.unique(root.getcol("DATA_DESC_ID"))

        for field in fields:
            field_rows = root.query(f"FIELD_ID=={int(field)}", style="")
            try:
                if field_rows.nrows() == 0:
                    continue

                for data_description in data_descriptions:
                    rows = field_rows.query(
                        f"DATA_DESC_ID=={int(data_description)}", style=""
                    )
                    try:
                        if rows.nrows() == 0:
                            continue

                        dd_table = table(f"{msname}/DATA_DESCRIPTION", ack=False)
                        try:
                            spectral_window_id = dd_table.getcol("SPECTRAL_WINDOW_ID")[
                                data_description
                            ]
                            polarisation_id = dd_table.getcol("POLARIZATION_ID")[
                                data_description
                            ]
                        finally:
                            dd_table.close()

                        ms_vis = rows.getcol(datacolumn)
                        ms_flags = rows.getcol("FLAG")
                        ms_weight = rows.getcol("WEIGHT")
                        uvw = -rows.getcol("UVW")
                        antenna1 = rows.getcol("ANTENNA1")
                        antenna2 = rows.getcol("ANTENNA2")
                        integration_time = rows.getcol("INTERVAL")
                        time = rows.getcol("TIME") - integration_time / 2.0

                        spectral_window = table(f"{msname}/SPECTRAL_WINDOW", ack=False)
                        try:
                            frequency = np.asarray(
                                spectral_window.getcol("CHAN_FREQ")[spectral_window_id]
                            )
                            channel_bandwidth = np.asarray(
                                spectral_window.getcol("CHAN_WIDTH")[spectral_window_id]
                            )
                        finally:
                            spectral_window.close()

                        polarisation = table(f"{msname}/POLARIZATION", ack=False)
                        try:
                            polarisation_frame = _polarisation_frame_from_corr_type(
                                polarisation.getcol("CORR_TYPE")[polarisation_id]
                            )
                        finally:
                            polarisation.close()

                        antenna = table(f"{msname}/ANTENNA", ack=False)
                        try:
                            all_names = np.asarray(antenna.getcol("NAME"))
                            valid = all_names != ""
                            if not np.any(valid):
                                valid = np.ones(len(all_names), dtype=bool)
                                all_names = np.asarray(
                                    [f"ANT{index}" for index in range(len(all_names))]
                                )

                            antenna_map = np.full(len(all_names), -1, dtype=int)
                            antenna_map[valid] = np.arange(np.count_nonzero(valid))
                            names = all_names[valid]
                            mount = np.asarray(antenna.getcol("MOUNT"))[valid]
                            diameter = np.asarray(antenna.getcol("DISH_DIAMETER"))[
                                valid
                            ]
                            xyz = np.asarray(antenna.getcol("POSITION"))[valid]
                            offset = np.asarray(antenna.getcol("OFFSET"))[valid]
                            stations = np.asarray(antenna.getcol("STATION"))[valid]
                        finally:
                            antenna.close()

                        antenna1 = antenna_map[antenna1]
                        antenna2 = antenna_map[antenna2]
                        nants = len(names)
                        baselines = pd.MultiIndex.from_tuples(
                            list(generate_baselines(nants)),
                            names=("antenna1", "antenna2"),
                        )
                        location = EarthLocation(
                            x=Quantity(xyz[0][0], "m"),
                            y=Quantity(xyz[0][1], "m"),
                            z=Quantity(xyz[0][2], "m"),
                        )
                        configuration = Configuration.constructor(
                            name="",
                            location=location,
                            names=names,
                            xyz=xyz,
                            mount=mount,
                            frame="ITRF",
                            receptor_frame=ReceptorFrame("linear"),
                            diameter=diameter,
                            offset=offset,
                            stations=stations,
                        )

                        field_table = table(f"{msname}/FIELD", ack=False)
                        try:
                            phase_direction = field_table.getcol("PHASE_DIR")[
                                field, 0, :
                            ]
                            source = field_table.getcol("NAME")[field]
                        finally:
                            field_table.close()
                        phasecentre = SkyCoord(
                            ra=phase_direction[0] * u.rad,
                            dec=phase_direction[1] * u.rad,
                            frame="icrs",
                            equinox="J2000",
                        )

                        time_index_by_row = np.zeros_like(time, dtype=int)
                        last_time = time[0]
                        time_index = 0
                        for row, row_time in enumerate(time):
                            if row_time > last_time + 0.5 * integration_time[row]:
                                if row_time <= last_time:
                                    raise ValueError(
                                        "MS is not time-sorted and cannot be converted"
                                    )
                                time_index += 1
                                last_time = row_time
                            time_index_by_row[row] = time_index

                        ntimes = time_index + 1
                        nbaselines = len(baselines)
                        nchan = len(frequency)
                        npol = polarisation_frame.npol
                        visibility_data = np.zeros(
                            (ntimes, nbaselines, nchan, npol), dtype=complex
                        )
                        flags = np.zeros((ntimes, nbaselines, nchan, npol), dtype=int)
                        weights = np.zeros(
                            (ntimes, nbaselines, nchan, npol), dtype=float
                        )
                        visibility_uvw = np.zeros((ntimes, nbaselines, 3))
                        visibility_times = np.zeros(ntimes)
                        visibility_integration_time = np.zeros(ntimes)

                        for row in range(len(time)):
                            baseline = baselines.get_loc((antenna1[row], antenna2[row]))
                            time_index = time_index_by_row[row]
                            visibility_times[time_index] = time[row]
                            visibility_data[time_index, baseline] = ms_vis[row]
                            flags[time_index, baseline] = ms_flags[row].astype(int)
                            weights[time_index, baseline] = ms_weight[
                                row, np.newaxis, :
                            ]
                            visibility_uvw[time_index, baseline] = uvw[row]
                            visibility_integration_time[time_index] = integration_time[
                                row
                            ]

                        visibilities.append(
                            Visibility.constructor(
                                uvw=visibility_uvw,
                                baselines=baselines,
                                time=visibility_times,
                                frequency=frequency,
                                channel_bandwidth=channel_bandwidth,
                                vis=visibility_data,
                                flags=flags,
                                weight=weights,
                                integration_time=visibility_integration_time,
                                configuration=configuration,
                                phasecentre=phasecentre,
                                polarisation_frame=polarisation_frame,
                                source=source,
                                meta={
                                    "MSV2": {
                                        "FIELD_ID": int(field),
                                        "DATA_DESC_ID": int(data_description),
                                    }
                                },
                            )
                        )
                    finally:
                        rows.close()
            finally:
                field_rows.close()
    finally:
        root.close()

    return visibilities


def export_visibility_to_ms(msname, vis_list, source_name=None):
    """Minimal Visibility to MS converter

    The MS format is much more general than the SDP Visibility
    so we cut many corners. This requires casacore to be
    installed. If not an exception ModuleNotFoundError is raised.

    Write a list of Visibility's to a MS file, split by field and
    spectral window

    :param msname: File name of MS
    :param vis_list: list of Visibility
    :param source_name: Source name to use
    :param ack: Ask casacore to acknowledge each table operation
    :return:
    """
    # pylint: disable=import-outside-toplevel
    try:
        from .msv2fund import Antenna, Stand
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("casacore is not installed") from exc

    try:
        from . import msv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("cannot import msv2") from exc

    # Start the table
    tbl = msv2.Ms(
        msname,
        ref_time=0,
        source_name=source_name,
        frame=vis_list[0].configuration.attrs["frame"],
        if_delete=True,
    )
    for vis in vis_list:
        if source_name is None or source_name != vis.source:
            source_name = vis.source
        # Check polarisation

        if vis.visibility_acc.polarisation_frame.type == "linear":
            polarization = ["XX", "XY", "YX", "YY"]
        elif vis.visibility_acc.polarisation_frame.type == "linearFITS":
            polarization = ["XX", "YY", "XY", "YX"]
        elif vis.visibility_acc.polarisation_frame.type == "linearnp":
            polarization = ["XX", "YY"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesI":
            polarization = ["I"]
        elif vis.visibility_acc.polarisation_frame.type == "circular":
            polarization = ["RR", "RL", "LR", "LL"]
        elif vis.visibility_acc.polarisation_frame.type == "circularnp":
            polarization = ["RR", "LL"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesIQUV":
            polarization = ["I", "Q", "U", "V"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesIQ":
            polarization = ["I", "Q"]
        elif vis.visibility_acc.polarisation_frame.type == "stokesIV":
            polarization = ["I", "V"]
        else:
            raise ValueError(
                f"Unknown visibility polarisation"
                f" {vis.visibility_acc.polarisation_frame.type}"
            )

        tbl.set_stokes(polarization)
        tbl.set_frequency(vis["frequency"].data, vis["channel_bandwidth"].data)
        n_ant = len(vis.attrs["configuration"].xyz)

        antennas = []
        names = vis.configuration.names.data
        xyz = vis.configuration.xyz.data
        for i, name in enumerate(names):
            antennas.append(Antenna(i, Stand(name, xyz[i, 0], xyz[i, 1], xyz[i, 2])))

        # Set baselines and data
        bl_list = []
        antennas2 = antennas

        for a_1 in range(0, n_ant):
            for a_2 in range(a_1, n_ant):
                bl_list.append((antennas[a_1], antennas2[a_2]))

        tbl.set_geometry(vis.configuration, antennas)

        int_time = vis["integration_time"].data
        assert vis["integration_time"].data.shape == vis["time"].data.shape

        # Now easier since the Visibility is baseline oriented
        for ntime, time in enumerate(vis["time"]):
            for ipol, pol in enumerate(polarization):
                if int_time[ntime] is not None:
                    tbl.add_data_set(
                        time.data,
                        int_time[ntime],
                        bl_list,
                        vis["vis"].data[ntime, ..., ipol],
                        weights=vis["weight"].data[ntime, ..., ipol],
                        pol=pol,
                        source=source_name,
                        phasecentre=vis.phasecentre,
                        uvw=vis["uvw"].data[ntime, :, :],
                    )
                else:
                    tbl.add_data_set(
                        time.data,
                        0,
                        bl_list,
                        vis["vis"].data[ntime, ..., ipol],
                        weights=vis["weight"].data[ntime, ..., ipol],
                        pol=pol,
                        source=source_name,
                        phasecentre=vis.phasecentre,
                        uvw=vis["uvw"].data[ntime, :, :],
                    )
    tbl.write()
