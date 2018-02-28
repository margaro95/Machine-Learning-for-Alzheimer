#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Throw the classification for configuration 3.

This is just a silly script to automate the task of launching the complete
classification of QSM vs Control for all the bands.
"""
from numpy import*
from createDataset import*
from createTargets import*
from createVector import*
from classificators import*

g = {"bias": -1, "input_scaling": 3}
dataset = load("dataset_alzheimer.npy")
dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 38, 9)
(dataset, Seed) = nonlinear_expand(dataset, 200, None, g)
targets = createTargets_DCLvsControl(array([30*9, 30*9, 29*9]), 18, 9)
(final_score, roc_auc) = classify_pseudoinverse(dataset, targets)
print(final_score, roc_auc)
