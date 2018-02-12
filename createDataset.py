#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module creates the vectors dataset to be used in sklearn."""
from os import listdir
from numpy import asarray

from basicfunctions import toDCL, toQSM, toControl
from createVector import createVector


def createDataset():
    """Create the dataset for the ML algorithm.

    Every row constitutes a sample with features as columns. The first 30 rows
    are the DCL patients (dataset[:29,:]), the next 30 rows are the QSM
    patients (dataset[30:59,:]) and the last 29 patients are the control group
    (dataset[60:,:]). Patients are ordered the same way the function listdir
    orders them. Bands within each patient are ordered from Delta to HighGamma2
    """
    nbands = 9
    # npatients = 30 + 30 + 29  # DCL + QSM + Control
    # nfeatures = 4005

    toDCL()
    dataset = []
    for i in range(len(listdir())):
        for j in range(nbands):
            dataset.append(list(createVector(listdir()[i], j)))

    toQSM()
    for i in range(len(listdir())):
        for j in range(nbands):
            dataset.append(list(createVector(listdir()[i], j)))

    toControl()
    for i in range(len(listdir())):
        for j in range(nbands):
            dataset.append(list(createVector(listdir()[i], j)))

    dataset = asarray(dataset)
    return(dataset)
