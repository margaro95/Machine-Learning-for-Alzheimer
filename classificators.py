#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements the classification."""
from sklearn import linear_model
from sklearn.model_selection import KFold
from numpy import ones


def run_crossvalidation(func):
    """Decorate your classificator to run the crossvalidation.

    The wrapper asks the user how many samples belong to one patient in order
    to make all the samples belonging to one patient constitute the test data.
    The output of the function (final_score) is not obtained doing a mean of
    all scores. It is the factor given by the number of patients who scored
    more than 0.5 divided by the total number of patients.
    """
    def func_wrapper(dataset, targets):
        splits = int(input("How many patients are in the dataset?\n"))
        kf = KFold(n_splits=splits)
        scores = []
        print("This samples ------------------------------ are predicted this")
        for train_index, test_index in kf.split(dataset):
            (prediction, scored) = func(dataset[train_index],
                                        targets[train_index],
                                        dataset[test_index],
                                        targets[test_index])
            scores.append(scored)
            print(str(test_index) + " - " + str(prediction)+" - " +
                  "Scored " + str(scored))
        final_score = list(scores > ones(splits)*0.5).count(True) / splits
        return final_score
    return func_wrapper


@run_crossvalidation
def classify_logistic(train_dataset, train_targets, test_dataset, test_target):
    """Classify test via trained logistic regression model."""
    modelo = linear_model.LogisticRegression()
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    return(prediction, scored)
