"""
Synopsis: Evaluate embeddings on the task of detecting adversarial examples
Author: Morgane Goibert <m.goibert@criteo.com>,
        Thomas Ricatte <t.ricatte@criteo.com>,
        Elvis Dohmatob <e.dohmatob@criteo.com>
"""
import typing

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.svm import OneClassSVM, SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tda.embeddings import get_gram_matrix
from tda.protocol import score_with_confidence
from tda.embeddings import KernelType
from tda.embeddings.persistence_landscape import compute_persistence_images
from tda.tda_logging import get_logger

logger = get_logger("REFACTOR")


def evaluate_embeddings(kernel_type: KernelType,
                        embeddings_train: typing.List,
                        embeddings_test: typing.List,
                        all_adv_embeddings_train: typing.Dict,
                        all_adv_embeddings_test: typing.Dict,
                        param_space: typing.List,
                        supervised: bool=False,
                        random_state=None,
                        n_jobs: int=1,
                        detector: BaseEstimator=None,
                        cv=5) -> pd.DataFrame:
    """
    Evaluate embeddings on the task of detecting adversarial examples
    """
    logger.info("Computing scores for %s (supervised=%s)" % (kernel_type,
                                                             supervised))
    grid_features_train = None
    scores = []
    all_eps = all_adv_embeddings_train.keys()
    for eps in all_eps:
        # compute features
        merged_embeddings_test = list(embeddings_test) + \
                                 list(all_adv_embeddings_test[eps])
        merged_embeddings_train = list(embeddings_train)
        if supervised:
            merged_embeddings_train += list(all_adv_embeddings_train[eps])
        kwargs = {}
        if kernel_type == KernelType.SlicedWasserstein:
            kwargs["kernel"] = "precomputed"
            if grid_features_train is None:
                grid_features_train = get_gram_matrix(
                    kernel_type=kernel_type,
                    embeddings_in=merged_embeddings_train,
                    embeddings_out=None,
                    params=param_space)
            assert len(grid_features_train) == len(param_space)
            grid_features_test = get_gram_matrix(
                kernel_type=kernel_type,
                embeddings_in=merged_embeddings_test,
                embeddings_out=merged_embeddings_train,
                params=param_space)
        elif kernel_type == KernelType.PersistenceLandscape:
            if grid_features_train is None:
                grid_features_train = [compute_persistence_images(
                    merged_embeddings_train,
                    n_jobs=n_jobs,
                    flatten=True, **fixture)[0]
                    for fixture in param_space]
            grid_features_test = [compute_persistence_images(
                merged_embeddings_test, n_jobs=n_jobs, flatten=True,
                **fixture)[0] for fixture in param_space]
        else:
            raise NotImplementedError(kernel_type)
        assert len(grid_features_test) == len(param_space)

        # fit and score detector
        n_test_pos = len(embeddings_test)
        n_test_neg = len(all_adv_embeddings_test[eps])
        flags_test_true = np.concatenate((np.ones(n_test_pos),
                                          np.zeros(n_test_neg)))

        if supervised:
            n_train_pos = len(embeddings_train)
            n_train_neg = len(all_adv_embeddings_train[eps])
            flags_train_true = np.concatenate((np.ones(n_train_pos),
                                               np.zeros(n_train_neg)))

            # XXX we use logistic regression instead of 2-class SVM
            if detector is None:
                if kernel_type == KernelType.SlicedWasserstein:
                    detector = GridSearchCV(
                        SVC(**kwargs), cv=cv,
                        param_grid={"C": [.001, .01, .1, 1.]})
                else:
                    clf = LogisticRegressionCV(Cs=10, cv=cv,
                                               max_iter=1000,
                                               random_state=random_state,
                                               n_jobs=n_jobs)
                    detector = Pipeline([("preproc", StandardScaler()),
                                         ("detector", clf)])
        else:
            flags_train_true = None
            if detector is None:
                detector = OneClassSVM(nu=.1, **kwargs)

        # XXX the loop below can / should be run in parallel
        best_fixture = None
        for fixture, features_train, features_test in zip(param_space,
                                                          grid_features_train,
                                                          grid_features_test):
            detector.fit(features_train, y=flags_train_true)
            flags_test_pred = detector.decision_function(features_test)
            auc = score_with_confidence(roc_auc_score, flags_test_true,
                                        flags_test_pred, n_bootstraps=20,
                                        random_state=random_state)
            if best_fixture is None or auc.is_better_than(best_auc):
                best_fixture = fixture
                best_auc = auc
        best_auc = best_auc._asdict()
        line = {"eps": eps, "best_params": best_fixture,
                "kernel_type": str(kernel_type),
                "supervised": supervised}
        for what in ["lower_bound", "value", "upper_bound"]:
            line["auc.%s" % what] = best_auc[what]
        scores.append(line)
        logger.info("eps=%.2f, AUC=%.2f%%" % (eps, 100 * auc.value))

    return pd.DataFrame(scores)
