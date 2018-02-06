"""This module defines the readdata function."""
from scipy import io


def readdata(archivo):
    """Return a usable file in Python from MATLAB file."""
    data = io.loadmat(archivo)
    return data
