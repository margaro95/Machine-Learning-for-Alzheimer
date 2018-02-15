#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements the classification."""
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from numpy import ones, asarray
import matplotlib.pyplot as plt


def run_crossvalidation(func):
    """Decorate your classifier to run the crossvalidation.

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
        y_test = []
        y_score = []
        print("This samples ------------------------------ are predicted this")
        for train_index, test_index in kf.split(dataset):
            (prediction, scored, decision) = func(dataset[train_index],
                                                  targets[train_index],
                                                  dataset[test_index],
                                                  targets[test_index])
            scores.append(scored)
            y_test.append(targets[test_index][0])
            y_score.append(decision[0])
            print(str(test_index) + " - " + str(prediction)+" - " +
                  "Scored " + str(scored))
        final_score = list(scores > ones(splits)*0.5).count(True) / splits
        fpr, tpr, thresholds = roc_curve(asarray(y_test), asarray(y_score))
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        clasificador = str(input("What classifier is this?\n"))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label=clasificador+' ROC curve, AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for the {} classifier'.format(clasificador))
        plt.legend(loc="lower right")
        plt.show()
        return final_score
    return func_wrapper


@run_crossvalidation
def classify_logistic(train_dataset, train_targets, test_dataset, test_target):
    """Classify test via trained logistic regression model."""
    modelo = linear_model.LogisticRegression()
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    decision = modelo.decision_function(test_dataset)
    return(prediction, scored, decision)
