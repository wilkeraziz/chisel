"""
Levenshtein distance

    is a pretty bad MT evaluation metric,
    I've implemented it here as an illustration of how to use the framework ;)

It requires installing python-Levenshtein

    pip install python-Levenshtein

@author waziz
"""

import chisel.mteval as mteval
import numpy as np
from Levenshtein import distance as levdistance


class WrappedLevenshtein(mteval.EvaluationMetric):

    def __init__(self, alias):
        super(WrappedLevenshtein, self).__init__(alias)
        self.evidence_space_ = None
        self.hypothesis_space_ = None
        self.distances_ = None

    def configure(self, config):
        pass

    def prepare_decoding(self, src, evidence, hypotheses):
        self.evidence_space_ = evidence
        self.hypothesis_space_ = hypotheses
        self.distances_ = np.zeros((len(hypotheses), len(evidence)))

        for i, hyp in enumerate(hypotheses):
            for j, ref in enumerate(evidence):
                self.distances_[i][j] = levdistance(hyp.projection, ref.projection)

    def loss(self, c, r):
        return self.distances_[c, r]/len(self.evidence_space_[r].projection)

    def coloss(self, c):
        raise NotImplementedError('Levenshtein does not support consensus decoding')

    def cleanup(self):
        self.evidence_space_ = None
        self.hypothesis_space_ = None
        self.distances_ = None


def construct(alias):
    return WrappedLevenshtein(alias)

