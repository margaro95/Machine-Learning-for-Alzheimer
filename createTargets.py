#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module creates the targets to be used in sklearn."""
from numpy import zeros, ones, concatenate


def createTargets_DCLvsControl(Ningroup, configuration, nbands):
    """Create the target for the ML algorithm - DCL vs Control.

    DCL will be marked as [0] and Control as a [1]. This script can also be
    used to generate the targets for the QSM vs Control case, being QSM = [0]
    and Control = [1]. If you want the targets for the configurations 30 to 38,
    just specify them as if they were 10 to 18.
    """
    if configuration == 1:
        target0 = zeros((Ningroup[0]))
        target1 = ones((Ningroup[2]))
        target = concatenate((target0, target1))
        return(target)
    elif configuration in range(10, 19):
        band_choice1 = [configuration % 10 + nbands * i
                        for i in range(Ningroup[0] // nbands)]
        band_choice3 = [configuration % 10 + nbands * i
                        for i in range(Ningroup[2] // nbands)]
        target0 = zeros((Ningroup[0]))[band_choice1]
        target1 = ones((Ningroup[2]))[band_choice3]
        target = concatenate((target0, target1))
        return(target)


def createTargets_DCLvsControlandQSM():
    """Create the target for the ML algorithm - DCL vs Control & QSM.

    DCL will be marked as [0] and Control & QSM as a [1].
    """
    target0 = zeros((30*9))
    target1 = ones((59*9))
    target = concatenate((target0, target1))
    return(target)


def createTargets_DCLandQSMvsControl():
    """Create the target for the ML algorithm - DCL & QSM vs Control.

    DCL & QSM will be marked as [0] and Control as a [1].
    """
    target0 = zeros((60*9))
    target1 = ones((29*9))
    target = concatenate((target0, target1))
    return(target)


def createTargets_DCLvsQSMvsControl():
    """Create the target for the ML algorithm - DCL vs QSM vs Control.

    DCL will be marked as [0], QSM as a [1] and Control as a [2].
    """
    target0 = zeros((30*9))
    target1 = ones((30*9))
    target2 = ones((29*9)) + 1
    target = concatenate((target0, target1, target2))
    return(target)


def createTargets_DCLvsQSM(Ningroup, configuration, nbands):
    """Create the target for the ML algorithm - DCL vs QSM.

    DCL will be marked as [0] and QSM as a [1].
    """
    if configuration == 2:
        target0 = zeros((Ningroup[0]))
        target1 = ones((Ningroup[1]))
        target = concatenate((target0, target1))
        return(target)
    elif configuration in range(20, 29):
        band_choice1 = [configuration % 20 + nbands * i
                        for i in range(Ningroup[0] // nbands)]
        band_choice2 = [configuration % 20 + nbands * i
                        for i in range(Ningroup[1] // nbands)]
        target0 = zeros((Ningroup[0]))[band_choice1]
        target1 = ones((Ningroup[2]))[band_choice2]
        target = concatenate((target0, target1))
        return(target)
