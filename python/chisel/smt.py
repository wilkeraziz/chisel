"""
This module contains data structures useful for decoding in the context of SMT.
@author waziz
"""
__author__ = 'waziz'

import re
import itertools
from collections import defaultdict
from semiring import CountSemiring, ProbabilitySemiring
import numpy as np
from util import npvec2str, kv2str
from operator import xor


class SVector(object):
    """
    A sparse feature vector (more like a map).
    Each component (identified by an integer key) is associated with a vector of real-valued features.
    Every iterator produced by this class is sorted by key.

    Iterating with iteritems returns pairs (key, value) where the key is the integer
    identifier of the component and the value is a tuple of feature values.

    Iterating with __iter__ returns the feature values in order (as if this was a normal vector).
    """

    def __init__(self, vecstr=None, fpairs=None):
        if not xor(vecstr is None, fpairs is None):
            raise ValueError('Either vecstr or fpairs must be given')
        if fpairs is not None:
            self.fmap_ = defaultdict(None, fpairs)
        else:
            # disambiguate spaces (separate components by tabs)
            vecstr = re.sub(r' ([^ ]+=)', r'\t\1', vecstr)
            pairs = vecstr.split('\t')
            fmap = defaultdict()
            # parse each component's features
            for pair in pairs:
                strkey, strvalue = pair.split('=')
                strvalue = re.sub('[][()]', '', strvalue)
                if len(strvalue.split()) != 1:
                    raise ValueError('Something is wrong about this component: %s=%s' % (strkey, strvalue))
                fmap[strkey] = float(strvalue)
                #fmap[strkey] = np.array([float(v) for v in strvalue.split()])
            self.fmap_ = fmap
        
        self.keys_ = tuple(sorted(self.fmap_.iterkeys()))
        self.values_ = tuple(v for k, v in sorted(self.fmap_.iteritems(), key = lambda pair: pair[0]))
        

    @property
    def keys(self):
        return self.keys_

    @property
    def values(self):
        return self.values_

    def as_array(self, selection=None):
        if selection is None:
            return np.array(self.values_)
            #return np.array([v for v in itertools.chain(*self.values_)])
        return np.array([self.fmap_.get(name, 0.0) for name in selection])
        #return np.array([fvalue for fvalue in itertools.chain(*[self.fmap_.get(fname, [0.0]) for fname in selection])])

    def __getitem__(self, key):
        return self.fmap_[key]

    def get(self, key, default=0.0):
        return self.fmap_.get(key, default)

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

    #def n_features(self):
    #    return sum(len(vec) for vec in self.itervalues(self))

    def __str__(self):
        return ' '.join('{0}={1}'.format(k, v) for k, v in itertools.izip(self.keys_, self.values_))
    
    #def __str__(self):
    #    return ' '.join(('%s=%s' % (k, str(v[0]))) if len(v) == 1 else ('%s=(%s)' % (k, ' '.join(str(x) for x in v))) for k, v in self.iteritems())


class Yield(object):

    def __init__(self, str_yield):
        self.yield_ = str_yield
        self.tokens_ = tuple(str_yield.split())

    def __str__(self):
        return self.yield_

    def len(self):
        return len(self.tokens_)

    def __iter__(self):
        return iter(self.tokens_)

    def __getitem__(self, i):
        return self.tokens_[i]

    @property
    def leaves(self):
        return self.tokens_


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

    def __init__(self, tree, vector, count):  #, log_ur, importance):
        self.tree_ = tree
        self.vector_ = vector
        self.count_ = count
        #self.log_ur_ = log_ur
        #self.importance_ = importance

    @property
    def tree(self):
        return self.tree_

    @property
    def vector(self):
        return self.vector_

    @property
    def count(self):
        return self.count_

    @property
    def n(self):
        return self.count_

    #@property
    #def log_ur(self):
    #    return self.log_ur_

    #@property
    #def importance(self):
    #    return self.importance_

    def __hash__(self):
        return hash(self.tree_)

    def __eq__(self, other):
        return self.tree_ == other.tree_

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return '%s\t%s\t%s' % (self.count_, self.tree_, self.vector_)
        #return '%s\t%s\t%s\t%s' % (self.count_, self.log_ur_, self.tree_, self.vector_)


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

    #def importance(self, op=ProbabilitySemiring.sum):
    #    """
    #    Returns the total importance (as importance is a normalised estimate which already incorporates the count of the derivation, we can simply sum)
    #    """
    #    return reduce(op, (d.importance for d in self.derivations_))
    

def groupby(derivations, key):
    """group derivations by key (a function over derivations)"""
    key2group = defaultdict(list)
    [key2group[key(d)].append(d) for d in derivations]
    return [DerivationGroup(group) for group in key2group.itervalues()]


class Solution(object):
    """
    """

    def __init__(self, Dy, p, q):
        """
        :param Dy: the group of derivations yielding y
        :param p: the (normalised) target posterior p(y)
        :param q: the (normalised) instrumental posterior q(y)
        """
        self.Dy_ = Dy
        self.p_ = p
        self.q_ = q

    @property
    def Dy(self):
        return self.Dy_

    @property
    def p(self):
        return self.p_

    @property
    def q(self):
        return self.q_

    def __str__(self):
        return 'p={0} q={1} yield={2}'.format(self.p_, self.q_, self.Dy_.projection)

    def format_str(self, keys=['p', 'q', 'yield'], separator='\t', named=True):
        fields = [None] * len(keys)
        for i, key in enumerate(keys):
            value = None
            if key == 'p':
                value = kv2str('p', self.p_, named)
            elif key == 'q':
                value = kv2str('q', self.q_, named)
            elif key == 'yield':
                value = kv2str('yield', self.Dy_.projection, named)
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

    def format_str(self, keys=['p', 'q', 'yield'], separator='\t', named=True):
        return separator.join([kv2str('target', self.target_, named),
                               self.solution_.format_str(keys, separator, named)])
