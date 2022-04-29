import numpy as np

def intersect2D(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Find row intersection indices of the whole set between 2D np.ndarrays, a and b.
    This assumes that either a or b is a true subset of the other.
    Returns the indices of the bigger set as np.ndarray
    """
    a, b = a.copy(), b.copy()
    # swap "a" and "b" if necessary so that "b" is always supposed to be a subset of "a"
    if b.shape[0] > a.shape[0]:
        tmp = a
        a = b
        b = tmp
    
    nrows, ncols = a.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [a.dtype]}
    c = np.intersect1d(a.view(dtype), b.view(dtype), return_indices=True)
    
    a_idxs = c[1] # 0=values, 1=a_idxs, 2=b_idxs
    
    return a_idxs