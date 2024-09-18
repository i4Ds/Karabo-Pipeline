# pylint: disable=invalid-name, too-few-public-methods, too-many-arguments
# pylint: disable=too-many-instance-attributes, too-many-locals
# pylint: disable=duplicate-code
"""
Source: https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels
Copyright: SKAO
License: Apache License 2.0

MeasurementSets V2 Reference Codes Based on Python-casacore
"""

import logging
import re
from datetime import datetime
from functools import total_ordering

import numpy

from .msv2supp import NUMERIC_STOKES, STOKES_CODES, merge_baseline

log = logging.getLogger("data-models-logger")

__version__ = "0.1"
__revision__ = "$Rev$"
__all__ = [
    "STOKES_CODES",
    "NUMERIC_STOKES",
    "Stand",
    "Observatory",
    "Antenna",
    "BaseData",
]


# Note: Python 3 cmp has changed to total_ordering
@total_ordering
class Stand:
    """
    Object to store the information (location and ID) about a stand.
    Stores stand:
     * ID number (id)
     * Position relative to the center stake in meters (x,y,z)

    The x, y, and z positions can also be accessed through subscripts:
     Stand[0] = x
     Stand[1] = y
     Stand[2] = z
    """

    def __init__(self, stand_id, x, y, z):
        self.id = stand_id
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __lt__(self, y):
        return self.id < y.id

    def __le__(self, y):
        return self.id <= y.id

    def __eq__(self, y):
        return self.id == y.id

    def __gt__(self, y):
        return self.id > y.id

    def __ge__(self, y):
        return self.id >= y.id

    def __cmp__(self, y):
        if self.id > y.id:
            return 1
        if self.id < y.id:
            return -1

        return 0

    def __str__(self):
        return (
            f"Stand {self.id}:  " f"x={self.x:.2f} m, y={self.y:.2f}m, z={self.z:.2f}m"
        )

    def __reduce__(self):
        return (Stand, (self.id, self.x, self.y, self.z))

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y
        if key == 2:
            return self.z

        raise ValueError(f"Subscript {key} out of range")

    def __setitem__(self, key, value):
        if key == 0:
            self.x = float(value)
        if key == 1:
            self.y = float(value)
        if key == 2:
            self.z = float(value)

        raise ValueError(f"Subscript {key} out of range")

    def __add__(self, std):
        try:
            # If its a Stand instance, do this
            out = (self.x + std.x, self.y + std.y, self.z + std.z)
        except AttributeError:
            try:
                # Maybe it is a list/tuple, so do this
                out = (self.x + std[0], self.y + std[1], self.z + std[2])
            except TypeError:
                out = (self.x + std, self.y + std, self.z + std)

        return out

    def __sub__(self, std):
        try:
            # If its a Stand instance, do this
            out = (self.x - std.x, self.y - std.y, self.z - std.z)
        except AttributeError:
            try:
                # Maybe it is a list/tuple, so do this
                out = (self.x - std[0], self.y - std[1], self.z - std[2])
            except TypeError:
                out = (self.x - std, self.y - std, self.z - std)

        return out


class Observatory:
    """
    Observatory class.
    Contains the location of the observatory.
    """

    def __init__(self, name, lon, lat, alt):
        self.name = name
        self.lon = lon
        self.lat = lat
        self.alt = alt


class Antenna:
    """
    Object to store the information about an antenna.  Stores antenna:
     * ID number (id)
     * Stand instance the antenna is part of (stand)
     * Polarization (0 == N-S; pol)
     * Antenna vertical mis-alignment in degrees (theta)
     * Antenna rotation mis-alignment in degrees (phi)
     * Fee instance the antenna is attached to (fee)
     * Port of the FEE used for the antenna (feePort)
     * Cable instance used to connect the antenna (cable)

     Some arguments are designed for future extension

    """

    def __init__(
        self,
        ant_id,
        stand=None,
        pol=0,
        theta=0.0,
        phi=0.0,
        fee=None,
        feePort=1,
        cable=None,
    ):
        self.id = int(ant_id)

        if stand is None:
            self.stand = Stand(0, 0, 0, 0)
        else:
            self.stand = stand

        self.pol = int(pol)
        self.theta = float(theta)
        self.phi = float(phi)

        self.fee = fee
        self.feePort = feePort

        self.cable = cable

    def __str__(self):
        return (
            f"Antenna {self.id}: " f"stand={self.stand.id}, polarization={self.pol}; "
        )

    def __reduce__(self):
        return (
            Antenna,
            (
                self.id,
                self.stand,
                self.pol,
                self.theta,
                self.phi,
                self.fee,
                self.feePort,
                self.cable,
            ),
        )

    def __lt__(self, y):
        return self.id < y.id

    def __le__(self, y):
        return self.id <= y.id

    def __eq__(self, y):
        return self.id == y.id

    def __gt__(self, y):
        return self.id > y.id

    def __ge__(self, y):
        return self.id >= y.id

    def __cmp__(self, y):
        if self.id > y.id:
            return 1
        if self.id < y.id:
            return -1

        return 0


class Frequency:
    """
    Information about the frequency setup used in the file.
    """

    def __init__(self, bandFreq, channelWidth, bandwidth):
        self.id = 1
        self.bandFreq = bandFreq
        self.chWidth = channelWidth
        self.totalBW = bandwidth
        self.sideBand = 1
        self.baseBand = 0

    def __eq__(self, y):
        sID = (self.bandFreq, self.chWidth, self.totalBW)
        yID = (y.bandFreq, y.chWidth, y.totalBW)
        return sID == yID

    # Note: In Python 3 __cmp__method is deprecated.


# We do not actively do any comparison here anymore.
# Still keeping the methods in case we want to call them directly.
@total_ordering
class MS_UVData:
    """
    UV visibility data set for a given observation time.
    """

    def __init__(
        self,
        obstime,
        inttime,
        baselines,
        visibilities,
        flags,
        weights=None,
        pol=STOKES_CODES["XX"],
        source=None,
        phasecentre=None,
        uvw=None,
    ):
        self.obstime = obstime
        self.inttime = inttime
        self.baselines = baselines
        self.visibilities = visibilities
        self.flags = flags
        self.weights = weights
        self.pol = pol
        self.source = source
        self.phasecentre = phasecentre
        self.uvw = uvw

    def __lt__(self, y):
        sID = (self.obstime, abs(self.pol))
        yID = (y.obstime, abs(y.pol))
        return sID > yID

    def __le__(self, y):
        sID = (self.obstime, abs(self.pol))
        yID = (y.obstime, abs(y.pol))
        return sID <= yID

    def __ge__(self, y):
        sID = (self.obstime, abs(self.pol))
        yID = (y.obstime, abs(y.pol))
        return sID >= yID

    def __gt__(self, y):
        sID = (self.obstime, abs(self.pol))
        yID = (y.obstime, abs(y.pol))
        return sID < yID

    def __eq__(self, y):
        sID = (self.obstime, abs(self.pol))
        yID = (y.obstime, abs(y.pol))
        return sID == yID

    def __cmp__(self, y):
        """
        Function to sort the self.data list in order of time and then
        polarization code.
        """

        sID = (self.obstime, abs(self.pol))
        yID = (y.obstime, abs(y.pol))
        if sID > yID:
            return 1
        if sID < yID:
            return -1

        return 0

    def time(self):
        """
        Observation time
        """
        return self.obstime

    def get_uvw(self, HA, dec, obs):
        """
        Get UVW information.
        """
        Nbase = len(self.baselines)
        uvw = numpy.zeros((Nbase, 3), dtype=float)

        # Phase center coordinates
        # Convert numbers to radians and, for HA, hours to degrees
        HA2 = HA * 15.0 * numpy.pi / 180
        dec2 = dec * numpy.pi / 180
        lat2 = obs.location.geodetic[1].to("rad").value

        # Coordinate transformation matrices
        trans1 = numpy.array(
            [
                [0, -numpy.sin(lat2), numpy.cos(lat2)],
                [1, 0, 0],
                [0, numpy.cos(lat2), numpy.sin(lat2)],
            ]
        )
        trans2 = numpy.array(
            [
                [numpy.sin(HA2), numpy.cos(HA2), 0],
                [
                    -numpy.sin(dec2) * numpy.cos(HA2),
                    numpy.sin(dec2) * numpy.sin(HA2),
                    numpy.cos(dec2),
                ],
                [
                    numpy.cos(dec2) * numpy.cos(HA2),
                    -numpy.cos(dec2) * numpy.sin(HA2),
                    numpy.sin(dec2),
                ],
            ]
        )

        for i, (a1, a2) in enumerate(self.baselines):
            # Go from a east, north, up coordinate system to a
            # celestial equation, east, north celestial pole system
            xyzPrime = a1.stand - a2.stand
            xyz = trans1 @ numpy.array([[xyzPrime[0]], [xyzPrime[1]], [xyzPrime[2]]])

            # Go from CE, east, NCP to u, v, w
            temp = trans2 @ xyz
            uvw[i, :] = numpy.squeeze(temp)

        return uvw

    def argsort(self, mapper=None, shift=16):
        """
        Sort by ID
        """
        packed = []
        for a1, a2 in self.baselines:
            if mapper is None:
                s1, s2 = a1.stand.id, a2.stand.id
            else:
                s1, s2 = mapper.index(a1.stand.id), mapper.index(a2.stand.id)
            packed.append(merge_baseline(s1, s2, shift=shift))
        packed = numpy.array(packed, dtype=int)

        return numpy.argsort(packed)


class BaseData:
    """
    Base Data class: For an observation of interferometer,
    we should have: Antenna, Frequency, Visibility Funcation, UVW
    """

    _MAX_ANTS = 255
    _PACKING_BIT_SHIFT = 8
    _STOKES_CODES = STOKES_CODES

    class _Antenna:
        """
        Holds information describing the location and properties of
        an antenna.
        """

        def __init__(self, ant_id, x, y, z, bits=8, name=None):
            self.id = ant_id
            self.x = x
            self.y = y
            self.z = z
            self.levels = bits
            self.name = name

        def getName(self):
            """
            Get the names of antennas.
            """
            if self.name is None:
                if isinstance(self.id, str):
                    return self.id

                return f"AT{self.id:3d}"

            return self.name

    @staticmethod
    def parse_time(ref_time):
        """
        Given a time as either a integer, float, string, or datetime
        object, convert it to a string in the formation
        'YYYY-MM-DDTHH:MM:SS'.
        """

        # Valid time string (modulo the 'T')
        timeRE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?")

        if type(ref_time) in (int, float):
            refDateTime = datetime.utcfromtimestamp(ref_time)
            ref_time = refDateTime.strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(ref_time, datetime):
            ref_time = ref_time.strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(ref_time, str):
            # Make sure that the string times are of the correct format
            if re.match(timeRE, ref_time) is None:
                raise RuntimeError(f"Malformed date/time provided: {ref_time}")

            ref_time = ref_time.replace(" ", "T", 1)

        else:
            raise RuntimeError("Unknown time format provided.")

        return ref_time

    def __init__(
        self,
        filename,
        ref_time=0.0,
        source_name=None,
        frame="ITRF",
        verbose=False,
    ):
        # File-specific information
        self.filename = filename
        self.verbose = verbose
        self.site_config = None
        # Observatory-specific information
        self.siteName = "Unknown"
        self.frame = frame
        self.source_name = source_name

        # Observation-specific information
        self.ref_time = self.parse_time(ref_time)
        self.nant = 0
        self.nchan = 0
        self.nstokes = 0
        self.refVal = 0
        self.refPix = 0
        self.channel_width = 0

        # Parameters that store the meta-data and data
        self.array = []
        self.freq = []
        self.stokes = []
        self.data = []
        self.uvw_RASCIL = None

    def __enter__(self):
        return self

    def __exit__(self, tp, value, traceback):
        self.write()
        self.close()

    def set_stokes(self, polList):
        """
        Given a list of Stokes parameters, update the object's
        parameters.
        """

        for pol in polList:
            if isinstance(pol, str):
                numericPol = self._STOKES_CODES[pol.upper()]
            else:
                numericPol = pol

            if numericPol not in self.stokes:
                self.stokes.append(numericPol)

        # Sort into order of 'XX', 'YY', 'XY', and
        # 'YX' or 'I', 'Q', 'U', and 'V'
        self.stokes.sort()
        if self.stokes[0] < 0:
            self.stokes.reverse()

        self.nstokes = len(self.stokes)

    # pylint:disable=expression-not-assigned
    def set_frequency(self, freq, channel_width):
        """
        Given a numpy array of frequencies, set the relevant common
        observation parameters and add an entry to the self.freq list.
        """

        if self.nchan == 0:
            self.nchan = len(freq)
            self.refVal = freq[0]
            self.refPix = 1
            self.channel_width = channel_width[0]
            offset = 0.0
        else:
            assert len(freq) == self.nchan
            offset = freq[0] - self.refVal
            self.channel_width = channel_width[0]

        if self.nchan == 1:
            totalWidth = self.channel_width
        else:
            totalWidth = numpy.abs(freq[-1] - freq[0])

        # only append if the setup didn't exist
        freqSetup = Frequency(offset, self.channel_width, totalWidth)
        (self.freq.append(freqSetup) if freqSetup not in self.freq else self.freq)

    def add_data_set(
        self,
        obstime,
        inttime,
        baselines,
        visibilities,
        flags=None,
        weights=None,
        pol="XX",
        source=None,
    ):
        """
        Create a UVData object to store a collection of visibilities.

        """

        if isinstance(pol, str):
            numericPol = self._STOKES_CODES[pol.upper()]
        else:
            numericPol = pol

        self.data.append(
            MS_UVData(
                obstime,
                inttime,
                baselines,
                visibilities,
                flags,
                weights=weights,
                pol=numericPol,
                source=source,
            )
        )

    def write(self):
        """
        Fill in the file will all of the required supporting metadata.
        """

        raise NotImplementedError

    def close(self):
        """
        Close out the file.
        """
        self.close()
