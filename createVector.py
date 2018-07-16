#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module creates the basic input vector for the ML algorithm."""
from numpy import triu, ones, array, newaxis

from basicfunctions import readdata


def createVector(file, band, vector=False):
    """Create the feature input vector for the ML algorithm.

    The band input to the function should be 0 for "Delta" band,
    1 for "Theta" band, 2 for "Alpha", 3 for "Beta1", 4 for "Beta2",
    5 for "Beta", 6 for "Gamma", 7 for "HighGamma1" and 8 for "HighGamma2".

    The vector returned contains the plv_pca_aal column belonging to the band
    specified to the file. It has been reshaped to a one dimensional vector
    containing the elements on the upper triangular region of the plv_pca_aal
    arranged in rows.

    ¡¡CUIDADO!!
    Miguel, el vector que crea esta función está al revés que el que tú haces.
    Este primero mete la fila entera antes de pasar a la siguiente fila. El
    tuyo si no recuerdo mal metía las columnas enteras antes de pasar a la
    siguiente columna.
    """
    mask = triu(ones((90, 90), dtype=bool), 1)
    matrix = readdata(file)["plv"][0, band][4]
    if vector is True:
        vector = matrix[mask]
    else:
        vector = matrix
    return(vector)


def createVector_ernesto(file, band):
    """Create the feature input vector for the ML algorithm.

    The band input to the function should be 0 for "Delta" band,
    1 for "Theta" band, 2 for "Alpha", 3 for "Beta1", 4 for "Beta2",
    5 for "Beta", 6 for "Gamma", 7 for "HighGamma1" and 8 for "HighGamma2".

    The vector returned contains the plv_pca_aal column belonging to the band
    specified to the file. It has been reshaped to a one dimensional vector
    containing the elements on the upper triangular region of the plv_pca_aal
    arranged in rows.

    Only difference is that only these features recommended by Ernesto are in
    use (25-26-31-32-33-34-37-38-61-62-67-68-85-86) in Matlab indexing, and the
    same index minus one in Python.
    """
    mask = triu(ones((14, 14), dtype=bool), 1)
    matrix = readdata(file)["plv"][0, band][4]
    chosen_ftres = [24, 25, 30, 31, 32, 33, 36, 37, 60, 61, 66, 67, 84, 85]
    matrix = matrix[array(chosen_ftres)[:, newaxis], array(chosen_ftres)]
    vector = matrix[mask]
    return(vector)
