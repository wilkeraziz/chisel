__author__ = 'waziz'

"""
This module contains data structures useful for decoding in the context of SMT.
@author waziz
"""
import re
import itertools
from collections import defaultdict
from semiring import CountSemiring, MaxTimesSemiring, SumTimesSemiring, ProbabilitySemiring
import numpy as np
from util import npvec2str, kv2str


class SVector(object):
    """
    A sparse feature vector (more like a map).
    Each component (identified by an integer key) is associated with a vector of real-valued features.
    Every iterator produced by this class is sorted by key.

    Iterating with iteritems returns pairs (key, value) where the key is the integer
    identifier of the component and the value is a tuple of feature values.

    Iterating with __iter__ returns the feature values in order (as if this was a normal vector).
    """

    def __init__(self, vecstr):
        # disambiguate spaces (separate components by tabs)
        vecstr = re.sub(r' ([^ ]+=)', r'\t\1', vecstr)
        pairs = vecstr.split('\t')
        fmap = defaultdict()
        # parse each component's features
        for pair in pairs:
            strkey, strvalue = pair.split('=')
            strvalue = re.sub('[][()]', '', strvalue)
            fmap[strkey] = np.array([float(v) for v in strvalue.split()])
        self.fmap_ = fmap
        self.keys_ = tuple(sorted(fmap.iterkeys()))
        self.values_ = tuple(v for k, v in sorted(fmap.iteritems(), key = lambda pair : pair[0]))

    @property
    def keys(self):
        return self.keys_

    @property
    def values(self):
        return self.values_

    def as_array(self, selection = None):
        if selection is None:
            return np.array([v for v in itertools.chain(*self.values_)])
        return np.array([fvalue for fvalue in itertools.chain(*[self.fmap_.get(fname, [0.0]) for fname in selection])])

    def __getitem__(self, key):
        return self.fmap_[key]

    def __contains__(self, key):
        return key in self.fmap_

    def iteritems(self):
        return self.fmap_.iteritems()

    def iterkeys(self):
        return self.fmap_.iterkeys()

    def itervalues(self):
        return self.fmap_.itervalues()

    def n_components(self):
        return len(self.fmap_)

    def n_features(self):
        return sum(len(vec) for vec in self.itervalues(self))

    def __str__(self):
        return ' '.join(('%s=%s' % (k, str(v[0]))) if len(v) == 1 else ('%s=(%s)' % (k, ' '.join(str(x) for x in v))) for k, v in self.iteritems())


class Tree(object):
    """
    A tree-like structure that represents a sequence of steps (e.g. translation rules).
    """

    # TODO: parse a proper tree (cdec-style)
    def __init__(self, derivationstr):
        """
        Parses a Moses-style string, e.g. "a b |0-0| c |3-4| d |2-2|"
        """
        strsegments = re.findall(r'[|][0-9]+-[0-9]+[|]', derivationstr)
        alignment_pattern = re.compile(r'[|][0-9]+-[0-9]+[|]')
        offset, tgt, src = None, None, None
        leaves = []
        derivation = []
        for span in re.split(r' *([|][0-9]+-[0-9]+[|]) *', derivationstr):
            if span.strip() == '':
                continue
            if alignment_pattern.match(span) is not None:
                strpair = re.sub(r'[|]', '', span.strip()).split('-')
                src = (int(strpair[0]), int(strpair[1]))
            else:
                offset = len(leaves)
                tgt = tuple(span.split())
                leaves.extend(tgt)
            if offset is not None and tgt is not None and src is not None:
                derivation.append((offset, tgt, src))
                offset, tgt, src = None, None, None

        self.str_ = derivationstr
        self.tree_ = tuple(derivation)
        self.leaves_ = tuple(leaves)

    @property
    def tree(self): # TODO: structure it like a tree.
        """
        @return tuple of bi-spans
        a bi-span is itself a tuple (target offset, target words, source span)
        'source span' is a pair (from, to)
        """
        return self.tree_

    @property
    def leaves(self):
        """
        @return tuple of words
        """
        return self.leaves_

    @property
    def projection(self):
        return ' '.join(self.leaves_)

    @property
    def bracketed_projection(self):
        return self.str_

    def __hash__(self):
        return hash(self.str_)

    def __eq__(self, other):
        return self.str_ == other.str_

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return self.str_


class Derivation(object):
    """
    A weighted derivation sampled a number of times
    """

    def __init__(self, tree, vector, score, count, importance = 1.0):
        self.tree_ = tree
        self.vector_ = vector
        self.score_ = score
        self.count_ = count
        self.importance_ = importance

    @property
    def tree(self):
        return self.tree_

    @property
    def vector(self):
        return self.vector_

    @property
    def score(self):
        return self.score_

    @property
    def count(self):
        return self.count_

    @property
    def n(self):
        return self.count_

    @property
    def importance(self):
        return self.importance_

    @property
    def ur(self):
        return self.importance_

    def __hash__(self):
        return hash(self.tree_)

    def __eq__(self, other):
        return self.tree_ == other.tree_

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return '%s\t%s\t%s\t%s\t%s' % (self.count_, self.importance_, self.score_, self.tree_, self.vector_)


class DerivationGroup(object):
    """
    Groups a list of derivations sharing the same projection
    """

    def __init__(self, derivations):
        """
        @param derivations
        @param key is the group's unique key
        """
        if len(derivations) == 0:
            raise Exception('A group cannot be empty')
        # get the key that indexes the group
        self.derivations_ = []
        self.mykey_ = derivations[0].tree.projection
        # add derivations
        self.derivations_.extend(derivations)

    @property
    def key(self):
        return self.mykey_

    @property
    def leaves(self):
        return self.derivations_[0].tree.leaves

    @property
    def projection(self):
        return self.derivations_[0].tree.projection

    def __len__(self):
        return len(self.derivations_)

    def __iter__(self):
        return iter(self.derivations_)

    def count(self, op=CountSemiring.sum):
        """
        @param op represents sum in a given semiring (e.g. CountSemiring.sum)
        @return total count according to op
        """
        return reduce(op, (d.count for d in self.derivations_))


def groupby(derivations, key):
    """group derivations by key (a function over derivations)"""
    key2group = defaultdict(list)
    [key2group[key(d)].append(d) for d in derivations]
    return [DerivationGroup(group) for group in key2group.itervalues()]


class Solution(object):
    """
    """

    def __init__(self, Dy, f, g, p, q, n):
        """
        :param Dy: the group of derivations yielding y
        :param f: the target feature vector
        :param g: the instrumental feature vector
        :param p: the (normalised) target posterior p(y)
        :param q: the (normalised) instrumental posterior q(y)
        :param n: (unnormalised) number of times this solution was sampled from q
        """
        self.Dy_ = Dy
        self.f_ = f
        self.g_ = g
        self.p_ = p
        self.q_ = q
        self.n_ = n

    @property
    def Dy(self):
        return self.Dy_

    @property
    def f(self):
        return self.f_

    @property
    def g(self):
        return self.g_

    @property
    def p(self):
        return self.p_

    @property
    def q(self):
        return self.q_

    @property
    def n(self):
        return self.n_

    def __str__(self):
        return 'p={0} q={1} n={2} yield={3} f={4} g={5}'.format(self.p_,
                                                                self.q_,
                                                                self.n_,
                                                                self.Dy_.projection,
                                                                npvec2str(self.f_),
                                                                npvec2str(self.g_))

    def format_str(self, keys=['p', 'q', 'n', 'yield', 'f', 'g'],
                   separator='\t',
                   named=False,
                   fnames=None,
                   gnames=None):
        fields = [None] * len(keys)
        for i, key in enumerate(keys):
            value = None
            if key == 'p':
                value = kv2str('p', self.p_, named)
            elif key == 'q':
                value = kv2str('q', self.q_, named)
            elif key == 'n':
                value = kv2str('n', self.n_, named)
            elif key == 'yield':
                value = kv2str('yield', self.Dy_.projection, named)
            elif key == 'f':
                value = kv2str('f', npvec2str(self.f_, fnames), named)
            elif key == 'g':
                value = kv2str('g', npvec2str(self.g_, gnames), named)
            else:
                raise Exception('Unkonwn field: %s' % key)
            fields[i] = value
        return separator.join(fields)


class KBestSolution(object):
    """
    Represents a solution which is the k-th best.
    """

    def __init__(self, solution, target, k):
        """
        :param solution: a Solution object
        :param target: final score used for ranking
        :param k: the position in the k-best list
        """
        self.k_ = k
        self.target_ = target
        self.solution_ = solution

    @property
    def k(self):
        return self.k_

    @property
    def target(self):
        return self.target_

    @property
    def solution(self):
        return self.solution_

    def __str__(self):
        return 'target={0} {1}'.format(self.target_, self.solution_)

    def format_str(self, keys=['p', 'q', 'n', 'yield', 'f', 'g'],
                   separator='\t',
                   named=False,
                   fnames=None,
                   gnames=None):
        return separator.join([kv2str('target', self.target_, named),
                               self.solution_.format_str(keys, separator, named, fnames, gnames)])