#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module defines basic changing directory functions and reading files."""
from os import chdir
from scipy import io
from numpy.random import seed


def readdata(archivo):
    """Return a usable file in Python from MATLAB file."""
    data = io.loadmat(archivo)
    return data


def toDCL():
    """Change directory to DCL folder."""
    chdir("/media/marcos/Seagate Expansion Drive/IFISC/alzheimer_ml/"
          "conectividad fuentes/DCL")


def toQSM():
    """Change directory to QSM folder."""
    chdir("/media/marcos/Seagate Expansion Drive/IFISC/alzheimer_ml/"
          "conectividad fuentes/QSM")


def toControl():
    """Change directory to NoQSM (aka, "Control") folder."""
    chdir("/media/marcos/Seagate Expansion Drive/IFISC/alzheimer_ml/"
          "conectividad fuentes/No QSM")


def toProjectFolder():
    """Change directory to project folder."""
    chdir("/media/marcos/Datos/Atom_Git_Projects/"
          "Machine_Learning_for_Alzheimer")


def set_seed(Seed=None):
    """Set seed for random generators to custom value or to random value."""
    if Seed is None:
        from time import time
        Seed = int((time() * 10 ** 6) % 1234567)
        print("This is your seed in case you want to replicate "
              "results\n"+str(Seed))
        seed(Seed)
        return Seed
    else:
        print("This is your seed in case you want to replicate "
              "results\n"+str(Seed))
        seed(Seed)
        return Seed
