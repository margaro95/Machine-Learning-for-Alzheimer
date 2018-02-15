#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Throw the classification for configuration 16.

This is just a silly script to automate the task of launching the complete
classification of DCL vs Control for Gamma band (configuration 16).
"""
from numpy import*
from createDataset import*
from createTargets import*
from createVector import*
from classificators import*
dataset = load("dataset_alzheimer.npy")
dataset = addaptDataset(dataset, array([9*30, 9*30, 9*29]), 4005, 16, 9)
targets = createTargets_DCLvsControl(array([30*9, 30*9, 29*9]), 16, 9)
classify_logistic(dataset, targets)
