#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a grid search on the parameters of our classification.

The parameters that we will be varying are the bias and input_scaling,
having the density of the input sparse matrix fixed (0.5), as well as the
number of nodes (200).
"""
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from numpy import (load, array, vstack, hstack, linspace,
                   save, zeros, median, mean)  # , arange

from createDataset import nonlinear_expand, addaptDataset
from createTargets import createTargets_DCLvsControl
from classificators import classify_pseudoinverse, classify_logistic

dataset = load("dataset_alzheimer.npy")
src_dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 16, 9)
targets = createTargets_DCLvsControl(array([30*9, 30*9, 29*9]), 16, 9)
src_dataset = src_dataset - mean(src_dataset, axis=0)
pca = PCA()
src_dataset = pca.fit(src_dataset).transform(src_dataset)
##########################################################################
# src_dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 2, 9)
# targets = createTargets_DCLvsQSM(array([30*9, 30*9, 29*9]), 2, 9)

# src_dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 38, 9)
# targets = createTargets_DCLvsControl(array([30*9, 30*9, 29*9]), 18, 9)
##########################################################################

grid = {"bias": linspace(0, 1, 20), "input_scaling": linspace(0.125, 2, 20),
        "density": array([1]), "nodes": array([2000])}
#grid = {"bias": array([-2 - 7/9]), "input_scaling": array([0.5]),
#        "density": 0.75, "nodes": 1000}
point_grids = ParameterGrid(grid)
# HACK: Isn't really a better way to make a stack of arrays to an empty array?
seeds_log = auc_log = zeros(10)

for g in point_grids:
    i = 0
    seeds_log_in_g = auc_log_in_g = array([])
    while i != 10:
        (dataset, Seed) = nonlinear_expand(src_dataset, None, g)
        (final_score, roc_auc, y_score) = classify_logistic(dataset, targets)
        seeds_log_in_g = hstack((seeds_log_in_g, Seed))
        auc_log_in_g = hstack((auc_log_in_g, roc_auc))
        print(roc_auc)
        i += 1
    seeds_log = vstack((seeds_log, seeds_log_in_g))
    auc_log = vstack((auc_log, auc_log_in_g))
save("seeds_log.npy", seeds_log)
save("auc_log.npy", auc_log)
median_auc_log = median(auc_log, axis=1)
save("median_auc_log.npy", median_auc_log[1:].reshape((20, 20)))
