import numpy as np
import os
import sys
import pyproj
import scipy.spatial.transform as te
import glob

tel_param = {
    "alma": [-23.0234, -67.7538, 5050],
    "vla": [34.078749167, -107.6177275, 2120],
    "askap": [-26.6961, 116.6369, 125],
    "meerkat": [-30.72111, 21.4111, 1000],
    "mkatplus": [-30.72111, 21.4111, 1000],
    "ska1low": [-50, 0, 1],
    "WSRT": [52.9145, 6.6031, 16],
    "sma": [19.8242, -155.4781, 4080],
    "GMRT": [19.0919, 74.0506, 656],
    "pdbi": [44.6330, 5.8965, 2550],
    "vlba": [19.8016, 155.4556, 10],
    "aca": [-23.0234, -67.7538, 5050],
    "atca": [-30.31278, 149.56278, 273],
    "carma": [37.2385, -118.3041, 2196],
    "lofar": [52.9089, 6.8689, 15],
    "mwa": [-26.7033, 116.6711, 125],
    "ngvla": [34, -40, 1000],
    "ska1mid": [-30.72111, 21.4111, 1000],
}  # Lat, Lon, alt


def read_cfg(filename):
    """
    Input: cfg configuration file
    Output: ENU coordinates (x,y,z)
    """
    filecontent = open(filename, "r")
    lines = filecontent.read().split("\n")
    lines = list(filter(None, lines))
    i = 0
    x = [0] * len(lines)
    y = [0] * len(lines)
    z = [0] * len(lines)
    for l in lines:
        if l[0] != "#":
            line = l.replace("\t", " ").split(" ")
            line = [i for i in line if i != ""]
            x[i] = np.float(line[0])
            y[i] = np.float(line[1])
            z[i] = np.float(line[2])
        i = i + 1
    x = [i for i in x if i != 0]
    y = [i for i in y if i != 0]
    z = [i for i in z if i != 0]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return x, y, z


def convert_ecef2latlong(x, y, z):
    """
    Convert ECEF to LAT-LONG Coordinates
    Input: ECEF Coordinates (x,y,z)
    Output: longitude, latitute and altitude
    """
    ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
    lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
    lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
    return lon, lat, alt


def convert_ecef2enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF to ENU Coordinates
    Input: ECEF Coordinates, latitude-longitude and altitude(x,y,z),lat,long,alt
    Output: ENU Coordinates
    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    )
    x_org, y_org, z_org = transformer.transform(lon0, lat0, alt0, radians=False)
    vec = np.array([[x - x_org, y - y_org, z - z_org]]).T
    rot1 = te.Rotation.from_euler(
        "x", -(90 - lat0), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1
    rot3 = te.Rotation.from_euler(
        "z", -(90 + lon0), degrees=True
    ).as_matrix()  # angle*-1 : left handed *-1
    rotMatrix = rot1.dot(rot3)
    enu = rotMatrix.dot(vec).T.ravel()
    return enu.T.reshape(x.shape[0], 3)
