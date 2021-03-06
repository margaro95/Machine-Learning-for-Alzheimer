#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module implements the classification."""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from numpy import ones, array

import matplotlib.pyplot as plt
import pdb

from classification_models import PseudoInverseRegression, BigEnsemble


def run_crossvalidation_with_ROC(func):
    """Decorate your classifier to run the crossvalidation and plot ROC curve.

    The wrapper asks the user how many samples belong to one patient in order
    to make all the samples belonging to one patient constitute the test data.
    The output of the function (final_score) is not obtained doing a mean of
    all scores. It is the factor given by the number of patients who scored
    more than 0.5 divided by the total number of patients.
    This decorator also plots the ROC curve for the classifier.

    The ROC curve is best explained here:
    https://ifisc.uib-csic.es/media/publications/publication/YSaJSPqST-2EdKQ1O2_-Nw.pdf
    """
    def func_wrapper(dataset, targets):
        # splits = int(input("How many patients are in the dataset?\n"))
        splits = 59
        kf = KFold(n_splits=splits)
        scores = []
        y_test = []
        y_score = []
        print("This samples ------------------------------ are predicted this")
        for train_index, test_index in kf.split(dataset):
            (prediction, scored, probas) = func(dataset[train_index],
                                                targets[train_index],
                                                dataset[test_index],
                                                targets[test_index])
            scores.append(scored)
            y_test.append(list(targets[test_index]))
            y_score.append(list(probas[:, 1]))
            print(str(test_index) + " - " + str(prediction)+" - " +
                  "Scored " + str(scored))
        y_test = list(array(y_test).ravel())
        y_score = list(array(y_score).ravel())
        final_score = list(scores > ones(splits)*0.5).count(True) / splits
        #pdb.set_trace()
        fpr, tpr, thresholds = roc_curve(array(y_test), array(y_score))
        roc_auc = auc(fpr, tpr)
        # plt.figure()
        # lw = 2
        # a = str(input("What classifier is this?\n"))
        # b = str(input("What classifying task is this?\n"))
        # plt.plot(fpr, tpr, color='darkorange',
                 # lw=lw, label=a + ' ROC curve, AUC = %0.2f' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC curve for the {0} classifier on {1}'.format(a, b))
        # plt.legend(loc="lower right")
        # plt.show()
        return (final_score, roc_auc, y_score)
    return func_wrapper


@run_crossvalidation_with_ROC
def classify_logistic(train_dataset, train_targets, test_dataset, test_target):
    """Classify test via trained logistic regression model."""
    modelo = LogisticRegression()
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    probas = modelo.predict_proba(test_dataset)
    return(prediction, scored, probas)


@run_crossvalidation_with_ROC
def classify_pseudoinverse(train_dataset, train_targets, test_dataset,
                           test_target):
    """Classify test via trained Moore-Penrose pseudoinverse of state."""
    modelo = PseudoInverseRegression()
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    probas = modelo.predict_proba(test_dataset)
    return(prediction, scored, probas)


@run_crossvalidation_with_ROC
def classify_NN(train_dataset, train_targets, test_dataset,
                test_target):
    """Classify test via trained Moore-Penrose pseudoinverse of state."""
    modelo = KNeighborsClassifier(2)
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    probas = modelo.predict_proba(test_dataset)
    return(prediction, scored, probas)


@run_crossvalidation_with_ROC
def classify_randomforest(train_dataset, train_targets, test_dataset,
                          test_target):
    """Classify test via trained Moore-Penrose pseudoinverse of state.

    WATCH OUT NOT TO BOOTSTRAP THE DATA BEFORE USING THE CLASSIFIER.
    Don't ensemble the random forests.
    """
    modelo = RandomForestClassifier(max_depth=None, n_estimators=500,
                                    max_features=None, bootstrap=True)
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    probas = modelo.predict_proba(test_dataset)
    return(prediction, scored, probas)


@run_crossvalidation_with_ROC
def classify_adaboost(train_dataset, train_targets, test_dataset,
                      test_target):
    """Classify test via trained Moore-Penrose pseudoinverse of state."""
    modelo = AdaBoostClassifier()
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    probas = modelo.predict_proba(test_dataset)
    return(prediction, scored, probas)


@run_crossvalidation_with_ROC
def classify_naivebayes(train_dataset, train_targets, test_dataset,
                        test_target):
    """Classify test via trained Moore-Penrose pseudoinverse of state."""
    modelo = GaussianNB()
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    scored = modelo.score(test_dataset, test_target)
    probas = modelo.predict_proba(test_dataset)
    return(prediction, scored, probas)


@run_crossvalidation_with_ROC
def classify_bigensemble(train_dataset, train_targets, test_dataset,
                         test_target):
    """Classify test via an ensemble of classifiers."""
    modelo = BigEnsemble()
    modelo.fit(train_dataset, train_targets)
    prediction = modelo.predict(test_dataset)
    #scored = modelo.score(test_dataset, test_target)
    scored = 0
    probas = modelo.predict_proba(test_dataset)
    return(prediction, scored, probas)
