from astropy.wcs import WCS


def get_slices(wcs: WCS):
    slices = []
    for i in range(wcs.pixel_n_dim):
        if i == 0:
            slices.append("x")
        elif i == 1:
            slices.append("y")
        else:
            slices.append(0)
    return slices


class Font:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
