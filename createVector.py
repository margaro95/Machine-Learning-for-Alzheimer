"""This module creates the basic input vector for the ML algorithm."""
from numpy import triu, ones
from basicfunctions import readdata


def createVector(file, band):
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
    vector = matrix[mask]
    return(vector)
