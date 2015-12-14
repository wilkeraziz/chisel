import numpy as np
from collections import defaultdict, deque


def make_features(leaves, max_order=4):
    """
    Returns 
        1) a list (indexed by ngram length minus 1) of features counts (indexed by reversed ngrams)
        2) a list (indexed by ngram length minus 1) of total counts 
    """
    # count kgrams for k=1..n
    cdef libcpp.vector[libcpp.map[libcpp.vector,float]] ngram_counts(max_order)

    ngram_counts = [defaultdict(float) for _ in range(max_order)]
    totals = np.zeros(max_order, float)
    reversed_context = deque()
    for w in leaves:
        # left trim the context
        if len(reversed_context) == max_order:
            reversed_context.pop()
        # count ngrams ending in w
        ngram = deque([w])
        # a) process w
        totals[len(ngram) - 1] += 1
        ngram_counts[len(ngram) - 1][tuple(ngram)] += 1
        # b) process longer ngrams ending in w
        for h in reversed_context:
            ngram.appendleft(h)
            totals[len(ngram) - 1] += 1
            ngram_counts[len(ngram) - 1][tuple(ngram)] += 1
        # c) adds w to the context
        reversed_context.appendleft(w)
    return ngram_counts, totals


def update_dmax(dmax, d):
    """In analogy to +=, this computes max="""
    for k, v in d.iteritems():
        mv = dmax.get(k, None)
        if mv is None:
            dmax[k] = v
        elif mv < v:
            dmax[k] = max(v, mv)


def clip_counts(d, dmax):
    """In analogy to -=, this implements min="""
    for k, v in d.iteritems():
        mv = dmax.get(k, 0.0)
        if v > mv:
            d[k] = mv

def max_features(references, max_order=4):
    """
    Returns
        1) a list (indexed by ngram length minus 1) feature counts (indexed by reversed ngram) from all references.
        we only keep the maximum count per feature
    """
    max_counts = [defaultdict(float) for _ in range(max_order)]
    for ref in references:
        counts, _ = make_features(ref, max_order)
        for i in range(max_order):
            update_dmax(max_counts[i], counts[i])
    return max_counts

def closest(v, L):
    """
    Returns the element in L (a sorted numpy array) which is closest to v

    >>> R = np.array([9,3,6])
    >>> R.sort()
    >>> R
    array([3, 6, 9])
    >>> [closest(i, R) for i in range(1,12)]
    [3, 3, 3, 3, 6, 6, 6, 9, 9, 9, 9]
    """
    i = L.searchsorted(v)
    return L[-1 if i == len(L) else (0 if i == 0 else (i if v - L[i - 1] > L[i] - v else i - 1))]

def bleu(hyp, ref_lenghts, ref_counts, max_order):
    # hyp length
    h_length = len(hyp)
    # closest ref length
    r_length = closest(h_length, ref_lenghts)
    # compute the brevity penalty
    bp = 1.0 if h_length > r_length else np.exp(1 - float(r_length) / h_length)
    # get h counts
    h_counts, h_totals = make_features(hyp, max_order)
    # clip counts wrt refs and compute precision with +1 smoothing
    cc = np.zeros(max_order, float)
    for i in range(max_order):
        clip_counts(h_counts[i], ref_counts[i])
        cc[i] = np.sum(h_counts[i].values())
    # compute ngram precisions with +1 smoothing
    pn = (cc + 1.0) / (h_totals + 1.0)
    # return BLEU
    return bp * np.exp(1.0 / max_order * np.sum(np.log(pn)))


class TrainingBLEU(object):

    def __init__(self, refs, max_order=4):
        self._max_order = max_order
        self._max_counts = max_features(refs, max_order)
        self._ref_lengths = np.array([len(ref) for ref in refs], float)

    def loss(self, hyp):
        return 1.0 - bleu(hyp, self._ref_lengths, self._max_counts, self._max_order)

