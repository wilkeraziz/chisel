"""
Levenshtein distance

    is a terrible MT evaluation metric,
    I've implemented it here as an illustration of how to use the framework ;)

It requires installing python-Levenshtein

    pip install python-Levenshtein

@author waziz
"""

import chisel.mteval as mteval
import numpy as np
from Levenshtein import distance as levdistance


evidence_space = None
hypothesis_space = None
distances = None

@mteval.decoding
def suffstats(src, evidence, hypotheses, consensus=False):
    """
    Compute sufficient statistics for BLEU
    :param src:
    :param EmpiricalDistribution evidence:
    :param EmpiricalDistribution hypotheses:
    """
    global evidence_space
    global hypothesis_space
    global distances

    evidence_space = evidence
    hypothesis_space = hypotheses
    distances = np.zeros((len(hypotheses), len(evidence)))

    for i, hyp in enumerate(hypotheses):
        for j, ref in enumerate(evidence):
            distances[i][j] = levdistance(hyp.projection, ref.projection)



@mteval.compare
def levenshtein(c, r):
    """
    Compare candidate c to reference r
    :param int c:
    :param int r:
    :return: bleu score
    """
    if r is mteval.EXPECTED:
        raise NotImplementedError('Levenshtein does not support consensus decoding')
    else:
        return distances[c, r]/len(evidence_space[r].projection)


@mteval.assess
def levenshtein(c):
    """
    :type c: int
    """
    raise NotImplementedError('Levenshtein training is not yet supported')


@mteval.cleanup
def cleanup():
    global evidence_space
    global hypothesis_space
    global distances
    evidence_space = None
    hypothesis_space = None
    distances = None


@mteval.reset
def reset():
    pass
