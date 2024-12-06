"""
Synopsis: Utilties for computing things related to persistence landscapes
Author: Elvis Dohmatob <e.dohmatob@criteo.com>
"""

import numpy as np
from joblib import Parallel, delayed


def persistence_diagrams_to_images(dgms, resolution=(20, 20), bandwidth=1.,
                                   backend="gudhi", n_jobs=1, flatten=False,
                                   **kwargs):
    assert isinstance(resolution, (list, tuple))
    assert len(resolution) == 2

    # inf-imputation
    sup = -np.inf
    dgms = [np.asanyarray(dgm).copy() for dgm in dgms]
    for dgm in dgms:
        mask = np.isinf(dgm)
        sup = max(dgm[~mask].max(), sup)
    for dgm in dgms:
        mask = np.isinf(dgm)
        dgm[mask] = sup

    if backend == "persim":
        import persim
        extractor = persim.PersImage(pixels=resolution, spread=bandwidth,
                                     **kwargs)

    elif backend == "gudhi":
        from gudhi.representations.vector_methods import PersistenceImage
        extractor = PersistenceImage(resolution=resolution,
                                     bandwidth=bandwidth, **kwargs)
        extractor.fit(dgms)
    else:
        raise NotImplementedError(backend)
    persimgs = Parallel(n_jobs=n_jobs)(
        delayed(extractor.transform)([np.asanyarray(dgm)]) for dgm in dgms)
    if not flatten:
        persimgs = [np.reshape(persimg, resolution) for persimg in persimgs]
    else:
        persimgs = [np.ravel(persimg) for persimg in persimgs]
    return persimgs
