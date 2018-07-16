#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Throw the classification for configuration 16.

This is just a silly script to automate the task of launching the complete
classification of DCL vs Control for Gamma band (configuration 16).
"""
import numpy as np
from createDataset import *
from createTargets import *
from createVector import *
from classificators import *
from sklearn.decomposition import PCA
import pdb

# g = {"bias": array([-2 - 7/9]), "input_scaling": array([0.5]),
#      "density": 0.75, "nodes": 1000}
g = {"bias": array([0]), "input_scaling": array([0.631578947]),
     "density": 1, "nodes": 5000}
targets = createTargets_DCLvsControl(array([30*9, 30*9, 29*9]), 18, 9)
dataset = np.load("dataset_alzheimer_pandas.npy")
dataset = np.delete(dataset, range(30, 60), 0)
(dataset, Seed) = nonlinear_expand(dataset, None, g)
# dataset = dataset - np.mean(dataset, axis=0)
# pca = PCA()
# dataset = pca.fit(dataset).transform(dataset)
# (dataset, Seed) = nonlinear_expand(dataset, None, g)
(final_score, roc_auc, y_score) = classify_bigensemble(dataset, targets)
print(final_score, roc_auc)
