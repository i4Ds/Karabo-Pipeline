# pylint: disable=too-many-locals, too-many-arguments, too-many-statements
# pylint: disable=too-many-nested-blocks,too-many-branches
# pylint: disable=invalid-name, duplicate-code
"""
Base functions to create and export Visibility
from/into Measurement Set files.
They take definitions of columns from msv2.py
and interact with Casacore.
"""


def export_visibility_to_ms(msname, vis_list, source_name=None):
    """Minimal Visibility to MS converter

    The MS format is much more general than the RASCIL Visibility
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
