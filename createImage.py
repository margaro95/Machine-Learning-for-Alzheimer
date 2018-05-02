#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Create images from PLV data."""
from PIL import Image
import numpy as np
from basicfunctions import toProjectFolder
import os
from createVector import createVector


def createImage(file, band, w, h):
    """Save PLV images to png format and with just the necessary pixels."""
    matrix = createVector(file, band)
    vector = np.reshape(matrix, (w, h))
    img = Image.fromarray(vector, 'L')
    cwd = os.getcwd()
    toProjectFolder()
    os.chdir('Images_keras')
    img.save(file + '_' + str(band) + '.png')
    os.chdir(cwd)
