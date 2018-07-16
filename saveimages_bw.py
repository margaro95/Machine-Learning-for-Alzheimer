#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform the routine to save all images for all patients and bands.

The images considered are the ones corresponding to the PLV of the AAL.
Those matrices are upper-diagonal and consist of 4005 non-zero values.
So that these values can be better analyzed by an image based learning
algorithm, we will transform the original 90x90 images to 89x45 rectangular
images.
"""
from basicfunctions import toDCL, toQSM, toControl
from os import listdir
from createImage import createImage
import pdb

nbands = 9
h, w = 89, 45

toDCL()
for i in range(len(listdir())):
    for j in range(nbands):
        createImage(listdir()[i], j, h, w)
        pdb.set_trace()
toQSM()
for i in range(len(listdir())):
    for j in range(nbands):
        createImage(listdir()[i], j, h, w)

toControl()
for i in range(len(listdir())):
    for j in range(nbands):
        createImage(listdir()[i], j, h, w)
