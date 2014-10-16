"""
@author waziz
"""
from chisel.metric.bleu import BLEU
import math
import numpy as np


def MBR(E, metric, normalise=False):
    M = len(E)
    scores = np.array([0.0] * M)
    for h, hyp in enumerate(E):
        for r, ref in enumerate(E):
            score = metric.exact(rid=r, hid=h)
            scores[h] += score * E.p(r, normalise)
    return scores


def expected_bleu(samples, bleusuff, bleu=BLEU.ibm_bleu, importance=lambda sample: 1.0):
    """
    Computes the expected (exact) BLEU of each candidate.
    @param samples is the candidates (also the evidence set)
    @param ngramstats, countstats (see count_ngrams)
    @param n max ngram order
    @return a list of pairs of the kind (sample, expected bleu) sorted from best to worst
    """
    # size of the evidence set
    M = len(samples)
    G = [0.0] * M
    n = bleusuff.maxorder
    # compute the exact clipped counts (intersection between the candidate and each evidence)
    for h, d in enumerate(samples):
        for r in xrange(M):
            # compute BLEU
            b = bleu(r=bleusuff.length(r),
                    c=bleusuff.length(h),
                    cc=bleusuff.cc(h, r),
                    tc=bleusuff.tc(h),
                    n=bleusuff.maxorder)
            # accumulate gain
            G[h] += b * samples[r].normcount * importance(samples[r])
    return G


def expected_linear_bleu(samples, bleusuff, T = 1, p = 0.85, r = 0.7):
    """
    Computes the expected linear BLEU (see Tromble et al 2008) of each candidate.
    @param samples is the candidates (also the evidence set)
    @param bleusuff (see BLEUSufficientStatistics)
    @param T, p, r are language-pair-specific parameters (see Tromble et al 2008) informed by devsets
    @param n max ngram order
    @return a list of pairs of the kind (sample, expected linear bleu) sorted from best to worst
    """

    # From (Tromble et al, 2008)
    # maximise the expected gain
    # d* = argmax_{d' \in H} { \theta_0 |d'| + \sum_{w \in N} \theta_w count_w(d') p(w|E)}
    # d' is a hypothesis (a candidate)
    # H is the hypothesis space
    # E is the evidence set
    # N is the set of n-grams (1 <= n <= 4) in D_e
    # p(w|E) = Z(E_w)/Z(E) where E_w is the set of hypotheses in E which contain w
    # count_w(d') is the number of occurrences of w in d'
    # \theta_0 and \theta_w(cn) are the Taylor coefficients
    # taylor coefficients

    theta_0 = -1.0/T
    cn = [0] * (bleusuff.maxorder + 1)
    for k in range(1, bleusuff.maxorder + 1):
        cn[k] = T * p * math.pow(r, k - 1)
    theta_w = lambda ngram: 1.0/(4 * cn[len(ngram)])

    G = [0] * len(samples)
    ngramstats = bleusuff.ngramstats
    for h, d in enumerate(samples):
        gain = theta_0 * len(d.leaves) + sum(theta_w(w) * stats.counts[h] * stats.posterior for w, stats in ngramstats.iteritems())
        G[h] = gain
    return G
