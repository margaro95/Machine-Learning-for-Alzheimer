#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a grid search on the parameters of our classification.

The parameters that we will be varying are the bias and input_scaling,
having the density of the input sparse matrix fixed (0.5), as well as the
number of nodes (200).
"""
from sklearn.decomposition import PCA


def ensemble_build(ensembles):
    from sklearn.model_selection import ParameterGrid
    from numpy import (load, array, vstack, hstack, linspace,
                       save, zeros, median, newaxis, mean)  # , arange

    from createDataset import (addaptDataset, nonlinear_expand, random_subspace)
    from createTargets import createTargets_DCLvsControl
    from classificators import classify_pseudoinverse, classify_logistic
    import pdb

    dataset = load("dataset_alzheimer.npy")
    src_dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 16, 9)
    targets = createTargets_DCLvsControl(array([30*9, 30*9, 29*9]), 16, 9)
    dataset = dataset - mean(dataset, axis=0)
    pca = PCA()
    src_dataset = pca.fit(src_dataset).transform(src_dataset)
    ##########################################################################
    # src_dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 2, 9)
    # targets = createTargets_DCLvsQSM(array([30*9, 30*9, 29*9]), 2, 9)

    # src_dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 38, 9)
    # targets = createTargets_DCLvsControl(array([30*9, 30*9, 29*9]), 18, 9)
    ##########################################################################

    # grid = {"bias": array([-2 - 7/9]), "input_scaling": array([0.5]),
            # "density": 1, "nodes": 2000}
    grid = {"bias": linspace(0, 1, 20)[8], "input_scaling": linspace(0.125, 2, 20)[15],
            "density": 1, "nodes": 2000}
    point_grids = ParameterGrid(grid)
    # HACK: Isn't really a better way to make a stack of arrays to an empty array?
    seeds_log = auc_log = zeros(100)
    y_score_matrix = zeros(59)[:, newaxis]
    roc_auc_log = []

    for e in range(ensembles):
        sbspace_dataset = random_subspace(src_dataset)
        (dataset, Seed) = nonlinear_expand(sbspace_dataset, None, grid)
        (final_score, roc_auc, y_score) = classify_logistic(dataset, targets)
        print(final_score, roc_auc)
        roc_auc_log.append(roc_auc)
        #pdb.set_trace()
        y_score_matrix = hstack((y_score_matrix, array(y_score)[:, newaxis]))

    #pdb.set_trace()
    save("y_score_matrix.npy", y_score_matrix[:, 1:])
    return(y_score_matrix[:, 1:])
