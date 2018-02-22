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


def addaptDataset(src_dataset, Ningroup, nfeatures, configuration, nbands):
    """Addapts the size of the dataset for different classification tasks.

    Function inputs:

        - src_dataset: ndarray with samples in rows and features in columns.
        - Ningroup: nd array. number of samples in each group NOT PATIENTS.
                    Groups are DCL, QSM and Control, not patients.
        - nfeatures: number of features (columns) per sample (patient)
        - configuration: choose 1 if it's the DCL vs Control case, 2 if it's
                         DCl vs QSM and 3 if its QSM vs Control. 10 if it's
                         Delta DCL vs Delta Control, 11 if it's Theta DCL vs
                         Theta Control, and so on. 20 if it is Delta DCL vs
                         Delta QSM, 22 if it is Alpha DCL vs Alpha QSM, and so
                         on; and 33 if it is Beta1 QSM vs Beta1 Control...
                         You get the idea...
                         Check createVector.__doc__ for more information on how
                         are the bands ordered.
        For example:
                addaptDataset(full_dataset,array([9*30,9*30,9*29]),4005,1,9)
                gives the dataset for the DCL vs Control case.
    """
    if configuration == 1:
        group1 = arange(Ningroup[0])
        group3 = arange(Ningroup[0]+Ningroup[1],
                        Ningroup[0]+Ningroup[1]+Ningroup[2])
        dataset = src_dataset[concatenate((group1, group3))[:, newaxis],
                              arange(nfeatures)]

    elif configuration == 2:
        group1 = arange(Ningroup[0])
        group2 = arange(Ningroup[0], Ningroup[0]+Ningroup[1])
        dataset = src_dataset[concatenate((group1, group2))[:, newaxis],
                              arange(nfeatures)]

    elif configuration == 3:
        group2 = arange(Ningroup[0], Ningroup[0]+Ningroup[1])
        group3 = arange(Ningroup[0]+Ningroup[1],
                        Ningroup[0]+Ningroup[1]+Ningroup[2])
        dataset = src_dataset[concatenate((group2, group3))[:, newaxis],
                              arange(nfeatures)]

    elif configuration in range(10, 19):
        band_choice1 = [configuration % 10 + nbands * i
                        for i in range(Ningroup[0] // nbands)]
        band_choice3 = [configuration % 10 + nbands * i
                        for i in range(Ningroup[2] // nbands)]
        group1 = arange(Ningroup[0])[band_choice1]
        group3 = arange(Ningroup[0]+Ningroup[1],
                        Ningroup[0]+Ningroup[1]+Ningroup[2])[band_choice3]
        dataset = src_dataset[concatenate((group1, group3))[:, newaxis],
                              arange(nfeatures)]

    elif configuration in range(20, 29):
        band_choice1 = [configuration % 20 + nbands * i
                        for i in range(Ningroup[0] // nbands)]
        band_choice2 = [configuration % 20 + nbands * i
                        for i in range(Ningroup[1] // nbands)]
        group1 = arange(Ningroup[0])[band_choice1]
        group2 = arange(Ningroup[0], Ningroup[0]+Ningroup[1])[band_choice2]
        dataset = src_dataset[concatenate((group1, group2))[:, newaxis],
                              arange(nfeatures)]

    elif configuration in range(30, 39):
        band_choice2 = [configuration % 30 + nbands * i
                        for i in range(Ningroup[1] // nbands)]
        band_choice3 = [configuration % 30 + nbands * i
                        for i in range(Ningroup[2] // nbands)]
        group2 = arange(Ningroup[0], Ningroup[0]+Ningroup[1])[band_choice2]
        group3 = arange(Ningroup[0]+Ningroup[1],
                        Ningroup[0]+Ningroup[1]+Ningroup[2])[band_choice3]
        dataset = src_dataset[concatenate((group2, group3))[:, newaxis],
                              arange(nfeatures)]

    return(dataset)


def nonlinear_expand(src_dataset, n_out, seed, width=0.75, lag=1.25):
    """Expand your dataset nonlinearly.

    Uses the non-linearity tanh(x) on a RNN to expand the data.
    You can select the dimensionality of the output dataset.
    The lag is where you want to place the mean of the activation values.
    The width is the amplitude of 2*std(activation) you want the activations to
    have.
    """
    from numpy import matmul, size, tanh, std, mean, histogram
    from numpy.random import rand
    from basicfunctions import set_seed
    from sklearn.preprocessing import normalize
    import pdb
    set_seed(seed)
    input_weights = rand(size(src_dataset, 1), n_out)
    #pdb.set_trace()
    src_dataset = normalize(src_dataset)
    #pdb.set_trace()
    activation = matmul(src_dataset, input_weights)
    width = width / std(activation)
    activation = (activation - mean(activation)) * width + lag
    new_dataset = tanh(activation)
    return new_dataset
