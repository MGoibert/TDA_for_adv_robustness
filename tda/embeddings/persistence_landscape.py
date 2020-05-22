"""
Synopsis: Utilties for computing things related to persistence landscapes
Author: Elvis Dohmatob <e.dohmatob@criteo.com>
"""

import numpy as np
from joblib import Parallel, delayed
import persim


def compute_persistence_images(dgms, pixels=(20, 20), n_jobs=1, flatten=False,
                               **kwargs):
    """Convert persistence diagrams to intensity images."""
    # inf-imputation
    assert isinstance(pixels, (list, tuple))
    assert len(pixels) == 2
    sup = -np.inf
    dgms = [np.asanyarray(dgm).copy() for dgm in dgms]
    for dgm in dgms:
        mask = np.isinf(dgm)
        sup = max(dgm[~mask].max(), sup)
    for dgm in dgms:
        mask = np.isinf(dgm)
        dgm[mask] = sup

    # the actual transformation
    extractor = persim.PersImage(pixels, **kwargs)
    persimgs = Parallel(n_jobs=n_jobs)(
        delayed(extractor.transform)([dgm]) for dgm in dgms)
    persimgs = [stuff[0] for stuff in persimgs]
    if flatten:
        persimgs = [persimg.ravel() for persimg in persimgs]
    return persimgs, extractor
