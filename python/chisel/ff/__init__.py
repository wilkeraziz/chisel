"""
Decorators for feature function definitions.

Let's agree on some terminology. I'm going to call your module a `scorer`. 
A scorer module can define one or more `feature functions`.
A feature has a `name` and a `value`. Feature names are strings and feature values are real-valued numbers.

To create a new feature functions simply:
    1) create your scorer module (e.g. my_scorer.py) and import `chisel.ff`
    2) use @chisel.ff.configure in order to load pre-trained models and parameters
    3) use @chisel.ff.preprocess in order to pre-process an input segment (e.g. parse the input)
       in this case you probably want to use @chisel.ff.reset in order to reset your scorer to a null state after finishing decoding a sentence
    4) use @chisel.ff.suffstats in order to pre-process a translation (e.g. parse the output)
       in this case you probably want to use @chisel.ff.cleanup in order to cleanup after suffstats before receiving a new translation to be scored
    5) use @chisel.ff.dense to define a new feature which returns a single value 
       the decorated function's name will name the feature
    6) use @chisel.ff.features('f1', 'f2', ..., 'fn') to define a function which returns a list of n feature values
       the arguments of the decoration are the feature names
    7) use @chisel.ff.sparse to define sparse features
       the decorated function's name will prefix the features' names

@author waziz
"""
from hypothesis import Hypothesis

import logging
import sys
import os
import itertools
import importlib
import chisel.ffpp.manager as ffpp


_SINGLE_ = []  # (func, fname)
_MULTIPLE_DENSE_ = []  # (func, fnames)
_SPARSE_ = []  # (func, fprexis)
_CONFIGURE_ = []
_PREPROCESS_ = []
_RESET_ = []
_SUFFSTATS_ = []
_CLEANUP_ = []



# available decorators


def configure(func):
    """decorate func with ff.configure if you want to configure your ff when the decoder is loaded"""
    _CONFIGURE_.append(func)
    return func

def configure2(func):
    """decorate func with ff.configure if you want to configure your ff when the decoder is loaded"""
    return func


def preprocess(func):
    """decorate func with ff.preprocess if you want to pre-process
    the input sentence right before the decoding process starts"""
    _PREPROCESS_.append(func)
    return func


def reset(func):
    """decorate func with ff.reset in order to reset your scorer to a null state after completing a translation"""
    _RESET_.append(func)
    return func


def suffstats(func):
    """decorate func with ff.suffstats if you have several ffs that share some underlying computation
    func will be run once for each hypothesis before any actual scorer is called"""
    _SUFFSTATS_.append(func)
    return func


def cleanup(func):
    """decorate func with ff.cleanup if you need to cleanup
    (for instance because you use ff.suffstats) after your scorers"""
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


# The following functions are not supposed to be used as decorators

def load_scorers(scorers):
    for fname, fdef in scorers:
        if os.path.isfile(fdef):
            try:
                logging.info('Loading additional feature definitions from file %s', fdef)
                prefix = os.path.dirname(fdef)
                sys.path.append(prefix)
                __import__(os.path.basename(fdef).replace('.py', ''))
                sys.path.remove(prefix)
            except:
                logging.error('Could not load feature definitions from file %s', fdef)
        else:
            try:
                logging.info('Loading additional feature definitions from module %s', fdef)
                importlib.import_module(fdef)
            except:
                logging.error('Could not load feature defitions from module %s', fdef)


def configure_scorers(config):
    """configure scorer modules"""
    [func(config) for func in _CONFIGURE_]


def preprocess_input(segment):
    """preprocess input"""
    [func(segment) for func in _PREPROCESS_]
    ffpp.preprocess_input(segment)


def reset_scorers():
    """resets scorers to a null state"""
    [func() for func in _RESET_]
    ffpp.reset_scorers()


def compute_features(hyp):
    """
    compute_features(hypothesis) -> list of named feature values (i.e. pairs of the kind (fname, fvalue))
    """
    # 1) give scorers the chance to prepare some sufficient statistics
    [func(hyp) for func in _SUFFSTATS_]
    # 2) evaluate scorers
    pairs = []
    # a) all single features (return 1 real value)
    pairs.extend((fname, fvalue) for fname, fvalue in ((fname, func(hyp)) for func, fname in _SINGLE_) if fvalue)
    #  b) all multiple dense features (return a list of pairs (fname, fvalue))
    for func, fnames in _MULTIPLE_DENSE_:
        fvalues = func(hyp)
        assert len(fnames) == len(fvalues), 'expected %d features, found %d' % (len(fnames), len(fvalues))
        pairs.extend((fname, fvalue) for fname, fvalue in itertools.izip(fnames, fvalues) if fvalue)
    #  c) all sparse features (return a list of pair (fsuffix, fvalue))
    for func, fprefix in _SPARSE_:
        pairs.extend(('{0}_{1}'.format(fprefix, fsuffix), fvalue) for fsuffix, fvalue in func(hyp) if fvalue)
    # 3) give ffs the chance to clean up
    [func() for func in _CLEANUP_]

    return ffpp.compute_features(hyp, pairs)

