#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Create images from PLV data."""
from PIL import Image
import numpy as np
from basicfunctions import toProjectFolder
import os
from createVector import createVector
import pdb


def createImage(archivo, band, h, w):
    """Save PLV images to png format and with just the necessary pixels."""
    matrix = createVector(archivo, band)
    vector = np.reshape(matrix, (h, w))
    vector = np.vstack((np.zeros(w), vector, np.ones(w)))
    img = Image.fromarray(vector, 'L')
    #  img.show()
    cwd = os.getcwd()
    toProjectFolder()
    os.chdir('Images_keras')
    pixels = img.load()
    new_im = Image.new('L', (w, h))
    pixels_new = new_im.load()
    rows_to_remove = [0, 90]
    rows_removed = 0
    for y in range(img.size[1]):
        if y not in rows_to_remove:
            #pdb.set_trace()
            for x in range(new_im.size[0]):
                pixels_new[x, y - rows_removed] = pixels[x, y]
        else:
            rows_removed += 1
    #pdb.set_trace()
    new_im.save(archivo + '_' + str(band) + '.png')
    os.chdir(cwd)
