"""
@author waziz
"""

from collections import defaultdict, deque
from scipy.sparse import dok_matrix
import numpy as np


class NGramFactory(object):
    def __init__(self, max_order):
        self.max_order_ = max_order
        self.ngrams_ = [[] for _ in range(max_order + 1)]
        self.n2i_ = [defaultdict() for _ in range(max_order + 1)]

    def get(self, words):
        """get(words) -> NGram object"""
        key = tuple(words)
        order = len(words)
        nid = self.n2i_[order].get(key, None)
        if nid is None:
            nid = len(self.ngrams_[order])
            self.n2i_[order][key] = nid
            self.ngrams_[order].append(key)
        return order, nid

    def __getitem__(self, (order, nid)):
        """self[nid] -> NGram object"""
        return self.ngrams_[order][nid]

    def ngrams(self, order):
        return self.ngrams_[order]


def make_features(leaves, ngram_factory, max_order=4):
    # count kgrams for k=1..n
    ngram_counts = [defaultdict(float) for _ in range(max_order + 1)]
    reversed_context = deque()
    for w in leaves:
        # left trim the context
        if len(reversed_context) == max_order:
            reversed_context.pop()
        # count ngrams ending in w
        ngram = deque([w])
        # a) process w
        order, nid = ngram_factory.get(ngram)
        ngram_counts[order][nid] += 1
        # b) process longer ngrams ending in w
        for h in reversed_context:
            ngram.appendleft(h)
            order, nid = ngram_factory.get(ngram)
            ngram_counts[order][nid] += 1
        # c) adds w to the context
        reversed_context.appendleft(w)
    return ngram_counts


def make_dok(kvpairs, max_dim):
    dok = dok_matrix((max_dim, 1), dtype=float)
    for k, v in kvpairs:
        dok[k, 0] = v
    return dok


def precision_n(c_ngram_counts, r_ngram_counts, max_order):
    # precisions = np.array([0.0 for _ in range(max_order + 1)])
    return np.array(
        [c_ngram_counts[n].minimum(r_ngram_counts[n]).sum() / c_ngram_counts[n].sum() for n in range(1, max_order + 1)])
    #for n in range(1, max_order + 1):
    #    cn = c_ngram_counts[n]
    #    rn = r_ngram_counts[n]
    #    precisions[n] = cn.minimum(rn).sum()/cn.sum()
    #return precisions