#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module creates everything related to datasets to be used in sklearn."""
from os import listdir
from numpy import asarray, concatenate, arange, newaxis

from basicfunctions import toDCL, toQSM, toControl
from createVector import createVector


def createDataset():
    """Create the dataset for the ML algorithm.

    Every row constitutes a sample with features as columns. The first 30 rows
    are the DCL patients (dataset[:29,:]), the next 30 rows are the QSM
    patients (dataset[30:59,:]) and the last 29 patients are the control group
    (dataset[60:,:]). Patients are ordered the same way the function listdir
    orders them. Bands within each patient are ordered from Delta to HighGamma2
    TODO -> create datasets including just some of the bands
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


def addaptDataset(src_dataset, Ningroup, nfeatures, configuration):
    """Addapts the size of the dataset for different classification tasks.

    Function inputs:

        - src_dataset: ndarray with samples in rows and features in columns.
        - Ningroup: nd array. number of samples in each group.
                    Groups are DCL, QSM and Control, not patients.
        - nfeatures: number of features (columns) per sample (patient)
        - configuration: choose 0 if its the DCL vs Control case, 1 if its
                         DCl vs QSM and 2 if its QSM vs Control (TODO ->
                         develop more configurations.)
        For example: addaptDataset(full_dataset,array([9*30,9*30,9*29]),4005,0)
        gives the dataset for the DCL vs Control case.
    """
    if configuration == 0:
        group1 = arange(Ningroup[0])
        group3 = arange(Ningroup[0]+Ningroup[1],
                        Ningroup[0]+Ningroup[1]+Ningroup[2])
        dataset = src_dataset[concatenate((group1, group3))[:, newaxis],
                              arange(nfeatures)]
    elif configuration == 1:
        group1 = arange(Ningroup[0])
        group2 = arange(Ningroup[0], Ningroup[0]+Ningroup[1])
        dataset = src_dataset[concatenate((group1, group2))[:, newaxis],
                              arange(nfeatures)]
    elif configuration == 2:
        group2 = arange(Ningroup[0], Ningroup[0]+Ningroup[1])
        group3 = arange(Ningroup[0]+Ningroup[1],
                        Ningroup[0]+Ningroup[1]+Ningroup[2])
        dataset = src_dataset[concatenate((group1, group2))[:, newaxis],
                              arange(nfeatures)]
    return(dataset)
