__author__ = 'waziz'

import numpy as np


def MAP(empdist, normalise=False):
    """Returns the (normalised) posterior based on the given empirical distribution"""
    return empdist.copy_posterior(normalise)
