import h5py as h5
import healpy as hp
import numpy as np


def h5_diter(g, prefix=""):
    """
    Get the data elements from the hdf5 datasets and groups
    Input: HDF5 file
    Output: Items and its path of data elements
    """
    for key, item in g.items():
        path = "{}/{}".format(prefix, key)
        if isinstance(item, h5.Dataset):  # test for dataset
            yield (path, item)
        elif isinstance(item, h5.Group):  # test for group (go down)
            yield from h5_diter(item, path)


def print_hd5_object_and_keys(hdffile):
    """
    Read HDF5 file
    Returns: HDF Object, relavent keys
    """
    with h5.File(hdffile, "r") as f:
        for (path, dset) in h5_diter(f):
            print(path)
    return f, f.keys()


def get_healpix_image(hdffile):
    """
    Get index maps, maps and frequency from HDF5 file
    """
    with h5.File(hdffile, "r") as f:
        for (path, dset) in h5_diter(f):
            pass
        print(f.keys())
        mapp = f["map"][:]
        # imapp = f['index_map'][:]
        # freq = f['index_map/freq'][:]
    return mapp


def get_vis_from_hdf5(hdffile):
    """
    Get index maps, maps and frequency from HDF5 file
    """
    with h5.File(hdffile, "r") as f:
        for (path, dset) in h5_diter(f):
            pass
        print(f.keys())
        vis = f["vis"][:]
    return vis


def convert_healpix_2_radec(arr):
    """
    Convert array from healpix to 2-D array of RADEC
    :param arr:
    :return: RADEC in degrees
    """
    nside = int(np.sqrt(arr.shape[0] / 12.0))
    index = np.arange(arr.shape[0])
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5 * np.pi - theta)
    return ra, dec, nside
