from astropy.wcs import WCS


def get_slices(wcs: WCS):
    slices = []
    for i in range(wcs.pixel_n_dim):
        if i == 0:
            slices.append('x')
        elif i == 1:
            slices.append('y')
        else:
            slices.append(0)
    return slices
