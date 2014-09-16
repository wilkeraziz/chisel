"""
Decorators for feature function definitions.
To create a new feature function simply:
    1) import ff
    2) use @configure in order to load pre-trained models and parameters
    3) use @feature to define a new feature which returns a single value 
    4) use @features('f1', 'f2', ..., 'fn') to define a function which returns a list of n feature values
    the arguments of the decoration are the feature names

@author waziz
"""
import logging
import sys
import os
import itertools

_SINGLE_ = [] # (func, fname)
_MULTIPLE_DENSE_ = [] # (func, fnames)
_SPARSE_ = [] # (func, fprexis)
_CONFIGURE_ = [] 
_SUFFSTATS_ = [] 
_CLEANUP_ = []

def configure(func):
    """decorate func with ff.configure if you want to configure your ff"""
    _CONFIGURE_.append(func)
    return func

def suffstats(func):
    """decorate func with ff.suffstats if you have several ffs that share some underlying computation
    func will be run once for each hypothesis before any actual scorer is called"""
    _SUFFSTATS_.append(func)
    return func

def cleanup(func):
    """decorate func with ff.cleanup if you need to cleanup (for instance because you use ff.suffstats) after your scorers"""
    _CLEANUP_.append(func)
    return func

def dense(scorer):
    """
    Decorating your scorer (scorer) with ff.dense will declare a dense feature identified by the scorer's name.
    IMPORTANT: scorer must return a SINGLE real value
    """
    _SINGLE_.append((scorer, scorer.__name__))
    logging.info('[dense] scorer/feature %s', scorer.__name__)
    return scorer

class features(object):
    """
    Decorating your scorer (scorer) with ff.features(list of names) will declare as many dense feature as you 
    declare through the arguments of the decorator.
    IMPORTANT: scorer must return a LIST (or tuple) of real values containing as many values as you declared.
    """

    def __init__(self, *args):
        self.fnames_ = list(args)
        if not self.fnames_:
            raise Exception('@ff.features requires at least one feature to be declared')

    def __call__(self, scorer):
        if not self.fnames_:
            self.fnames_ = [scorer.__name__]
        logging.info('[dense] scorer %s features %s', scorer.__name__, str(self.fnames_))
        _MULTIPLE_DENSE_.append((scorer, self.fnames_))
        return scorer

def sparse(scorer):
    """
    Decorating your scorer (scorer) with ff.sparse will declare a template for sparse features, prefixed by the scorer's name.
    IMPORTANT: scorer must return a LIST (or tuple) of named features
    named features are pairs of the kind (suffix, value). 
    chisel doesn't care what `suffix` is as long as it can be converted into a string (without any white spaces).
    Each sparse feature will be named prefix_suffix where prefix is the scorer's name.
    """
    _SPARSE_.append((scorer, scorer.__name__))
    logging.info('[sparse] scorer/feature %s', scorer.__name__)
    return scorer

def load_features(features):
    for featdef in features:
        logging.info('Loading additional feature definitions from %s', featdef)
        prefix = os.path.dirname(featdef)
        sys.path.append(prefix)
        __import__(os.path.basename(featdef).replace('.py', ''))
        sys.path.remove(prefix)

def configure_features(config):
    [func(config) for func in _CONFIGURE_]

def compute_features(hypothesis):
    """
    compute_features(hypothesis) -> list of named feature values (i.e. pairs of the kind (fname, fvalue))
    """
    # 1) give scores the chance to prepare some sufficient statistics
    [func(hypothesis) for func in _SUFFSTATS_]
    # 2) evaluate scorers
    pairs = []
    #  a) all single features (return 1 real value)
    pairs.extend((fname, func(hypothesis)) for func, fname in _SINGLE_)
    #  b) all multiple dense features (return a list of pairs (fname, fvalue))
    for func, fnames in _MULTIPLE_DENSE_:
        fvalues = func(hypothesis)
        assert len(fnames) == len(fvalues), 'expected %d features, found %d' % (len(fnames), len(fvalues))
        pairs.extend(itertools.izip(fnames, fvalues))
    #  c) all sparse features (return a list of pair (fsuffix, fvalue))
    for func, fprefix in _SPARSE_:
        pairs.extend(('{0}_{1}'.format(fprefix, fsuffix), fvalue) for fsuffix, fvalue in func(hypothesis))
    # 3) give ffs the chance to clean up
    [func() for func in _CLEANUP_]
    
    return pairs
