#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains classification models not included in sklearn."""
from sklearn.utils.extmath import softmax
from sklearn.linear_model.base import LinearClassifierMixin
from numpy.linalg import pinv
from scipy.linalg import pinv2
from numpy import (hstack, ones, size, dot, transpose, delete, c_, newaxis,
                   unique)
from sklearn.exceptions import NotFittedError


import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import time


class BigEnsemble(BaseEstimator):
    def __init__(self):
        self.clfs_list = [LogisticRegression(penalty='l2', C=1.0),
                          LogisticRegression(penalty='l2', C=100.0),
                          LogisticRegression(penalty='l1'),
                          RandomForestClassifier(max_depth=20, n_jobs=4),
                          GradientBoostingClassifier(learning_rate=0.1, n_estimators=50),
                          GradientBoostingClassifier(loss="exponential", learning_rate=0.01, n_estimators=80),
                          SVC(probability=True),
                          SVC(C=1e2, probability=True)]

        self.pipes_list = [make_pipeline(StandardScaler(),
                           reg) for reg in self.clfs_list]

        self.meta_clf = LogisticRegression(C=1.)

    def _fit_pipes(self, X_train, y_train):
        for clf in self.pipes_list:
            print("Fitting ", clf, end="")
            start = time.time()
            clf.fit(X_train, y_train)
            print(", Done in %.3f min" % ((time.time()-start)/60))

    def _predict_pipes(self, X_validation):
        res = []
        for clf in self.pipes_list:
            res.append(clf.predict_proba(X_validation))
    	# Lo que hago concatenando así es coger y poner todos los resultados como
    	# si fuera un nuevo dataset (algo así):
    	# 		       clf1_label1   clf1_label2    clf2_label1    ...
    	# sample_val1	   0.9		     0.1		 0.85	       ...
    	# sample_val2	   0.2		     0.8		 0.1	       ...
    	# sample_val3	   0.1		     0.9		 0.15	       ...
    	# sample_val4	   0.99		     0.01		 0.9	       ...
        return np.concatenate(res, axis=1)


    def fit(self, X, y):
        # Validación en local
        # train_index, validation_index = KFold(range(y.size),
        #                                                  test_size=0.37,
        #                                                  shuffle=True,
        #                                                  random_state=42)
        # X_train = X.iloc[train_index]
        # X_validation = X.iloc[validation_index]
        # y_train = y[train_index]
        # y_validation = y[validation_index]
        #
        # self._fit_pipes(X_train, y_train)
        #
        # y_predicted = self._predict_pipes(X_validation)
        self._fit_pipes(X, y)
        y_predicted = self._predict_pipes(X)
# Fin de la validación en local
        # self.meta_clf.fit(y_predicted, y_validation)
        self.meta_clf.fit(y_predicted, y)
        return self

    def predict(self, X):
        y_predicted = self._predict_pipes(X)

        return self.meta_clf.predict(y_predicted)

    def predict_proba(self, X):
        y_predicted = self._predict_pipes(X)

        return self.meta_clf.predict_proba(y_predicted)


class PseudoInverseRegression(LinearClassifierMixin):
    """Pseudo Inverse Regression classifier.

    This is a classifier in the sklearn.linear_model flavour.
    """

    def fit(self, X, y):
        """Fit your data and create the output vector."""
        bias = ones((size(X, 0), 1))
        X = hstack((bias, X))
        W = dot(pinv2(X), y)
        self.classes_ = unique(y)
        self.coef_ = transpose(delete(W, 0, 0))[newaxis]
        self.intercept_ = W[0]
        return self

    def predict_proba(self, X):
        """Predict probabilities for the (test) dataset."""
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")
        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for multi_class="multinomial" and binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision_2d = c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d, copy=False)
