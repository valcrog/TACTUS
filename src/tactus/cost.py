import numpy as np
import scipy

def compute_cost_matrix(X, Y, metric='euclidean'):
    """Compute the cost matrix of two feature sequences

    Args:
        X (np.ndarray): Sequence 1
        Y (np.ndarray): Sequence 2
        metric (str): Cost metric, a valid strings for scipy.spatial.distance.cdist (Default value = 'euclidean')

    Returns:
        C (np.ndarray): Cost matrix
        
    ---
    Source : 
    This function comes from the library `libfmp`, part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP) 
    Link : https://github.com/meinardmueller/libfmp
    Licence : The MIT license
    """
    X, Y = np.atleast_2d(X, Y)
    C = scipy.spatial.distance.cdist(X.T, Y.T, metric=metric)
    return C