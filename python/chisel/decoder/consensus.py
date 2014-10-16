"""
@author waziz
"""
from chisel.metric.bleu import BLEU
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


def consensus(E, metric, normalise=False):
    scores = [metric.expected(i, normalise) for i, Dy in enumerate(E)]
    return scores


def consensus_bleu(samples, bleusuff, bleu=BLEU.ibm_bleu):
    ngramstats = bleusuff.ngramstats
    # TODO: wrap in a class that prepares the expectations
    N = [w for w in ngramstats.iterkeys()]
    C = csc_matrix([stats.counts for stats in ngramstats.itervalues()])
    L = np.matrix([bleusuff.length(h) for h in xrange(len(samples))])
    P = np.matrix([sample.normcount for sample in samples]).transpose()
    Ec = C * P
    El = L * P
    N2E = {w: Ec[i] for i, w in enumerate(N)}


    S = [0] * len(samples)
    for h, sample in enumerate(samples):
    
        # clip to the expected counts
        cc = [0] * (bleusuff.maxorder + 1)
        for w, c in bleusuff.ngrams(h).iteritems():
            cc[len(w)] += min(c, N2E.get(w, 0))

        # compute BLEU
        S[h] = bleu(r = El[0,0], 
                c = bleusuff.length(h), 
                cc = cc,
                tc = bleusuff.tc(h),
                n = bleusuff.maxorder)

    return S


def cobleu(reference, samples, bleusuff, bleu = BLEU.ibm_bleu):
    """
    See Pauls et at 2009
    it is basically BLEU where the candidate is represented by a vector of expected counts.

    If BLEU(c, r) represents the modified ngram precision between a candidate c  and a reference r, then:
        * in CoBLEU training, r is the reference and c is represented by expected counts
            we then maximise theta (the parameters of the model)

        * in Consensus decoding, c is a hypothesis and r is represented by expected counts
    """
    pass
