import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import healpy as hp
import healpy.visufunc

def h5_diter(g, prefix=''):
    '''
    Get the data elements from the hdf5 datasets and groups
    Input: HDF5 file 
    Output: Items and its path of data elements
    '''
    for key, item in g.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5_diter(item, path)


def read_hd5(hdffile):
    '''
    Read HDF5 file
    Returns: HDF Object, relavent keys
    '''
    with h5.File(hdffile, 'r') as f:
        for (path, dset) in h5_diter(f):
    return f,f.keys()

def get_healpix_map(hdffile):
    '''
    Get index maps, maps and frequency from HDF5 file
    '''
    f=read_hd5(hdffile)
    mapp=f['map'];imapp=f['index_map'];freq=f['index_map/freq']
    return mapp,imapp,freq





