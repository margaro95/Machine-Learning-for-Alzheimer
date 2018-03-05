#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains classification models not included in sklearn."""
from sklearn.utils.extmath import softmax
from sklearn.linear_model.base import LinearClassifierMixin
from numpy.linalg import pinv
from numpy import (hstack, ones, size, dot, transpose, delete, c_, newaxis,
                   unique)
from sklearn.exceptions import NotFittedError


class PseudoInverseRegression(LinearClassifierMixin):
    """Pseudo Inverse Regression classifier.

    This is a classifier in the sklearn.linear_model flavour.
    """

    def fit(self, X, y):
        """Fit your data and create the output vector."""
        bias = ones((size(X, 0), 1))
        X = hstack((bias, X))
        W = dot(pinv(X), y)
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
