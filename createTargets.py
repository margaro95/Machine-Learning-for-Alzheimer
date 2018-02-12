#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This module creates the targets to be used in sklearn."""
from numpy import zeros, ones, concatenate


def createTargets_DCLvsControl():
    """Create the target for the ML algorithm - DCL vs Control.

    DCL will be marked as [0] and Control as a [1]. This script can also be
    used to generate the targets for the QSM vs Control case, being QSM = [0]
    and Control = [1].
    """
    target0 = zeros((30))
    target1 = ones((29))
    target = concatenate((target0, target1))
    return(target)


def createTargets_DCLvsControlandQSM():
    """Create the target for the ML algorithm - DCL vs Control & QSM.

    DCL will be marked as [0] and Control & QSM as a [1].
    """
    target0 = zeros((30))
    target1 = ones((59))
    target = concatenate((target0, target1))
    return(target)


def createTargets_DCLandQSMvsControl():
    """Create the target for the ML algorithm - DCL & QSM vs Control.

    DCL & QSM will be marked as [0] and Control as a [1].
    """
    target0 = zeros((60))
    target1 = ones((29))
    target = concatenate((target0, target1))
    return(target)


def createTargets_DCLvsQSMvsControl():
    """Create the target for the ML algorithm - DCL vs QSM vs Control.

    DCL will be marked as [0], QSM as a [1] and Control as a [2].
    """
    target0 = zeros((30))
    target1 = ones((30))
    target2 = ones((29)) + 1
    target = concatenate((target0, target1, target2))
    return(target)


def createTargets_DCLvsQSM():
    """Create the target for the ML algorithm - DCL vs QSM.

    DCL will be marked as [0] and QSM as a [1].
    """
    target0 = zeros((30))
    target1 = ones((30))
    target = concatenate((target0, target1))
    return(target)
