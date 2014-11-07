import logging
import os
import sys
import importlib

EXPECTED = None

_CONFIGURE_ = []
_PRECOMP_ = []
_TRAINING_ = []
_DECODING_ = []
_RESET_ = []
_CLEANUP_ = []

_COMPARE_ = {}
_ASSESS_ = {}


"""
Decoding:

    argmin          sum         L(y', y) p(y)
      y' in Yh        y in Ye


    Yh    HYP (Ye)       REF (Yh)
    1     y1,1...y1,N    y'1
    2     y2,1...y2,N    y'2
    ...   ...            ...
    M     yM,1...yM,N    y'M

Learning

    argmin      sum             sum        L(y', y)p(y)
      t,l        (x',y') in T     y in Ye

    T    HYP (Ye)        REF (T)
    1    y1,1...y1,N     y'1,1 y'1,2 ... y'1,r
    2    y2,1...y2,N     y'2,1 y'2,2 ... y'2,r
    ...  ...             ...
    M    yM,1...yM,N     y'M,1 y'M,2 ... y'1,r
"""


def configure(func):
    """decorate func with mteval.configure if you want to configure your mteval ***when the decoder is loaded***"""
    _CONFIGURE_.append(func)
    return func


def precomp(func):
    """
    Use this decorator to precompute stuff for an entire training set.
    func(training_set)
    """
    _PRECOMP_.append(func)


def training(func):
    """func(src, references, hypotheses, consensus=False)"""
    _TRAINING_.append(func)


def decoding(func):
    """func(src, evidence, hypotheses, consensus=False)"""
    _DECODING_.append(func)


def compare(func):
    """
    Use this decorator to compute a gain function for decoding
    :param func: func(c, r) -> float
    """
    logging.info('@chisel.mteval.compare %s', func.__name__)
    _COMPARE_[func.__name__] = func


def assess(func):
    """
    Use this decorator to compute a gain function for training
    :param func: func(c) -> float
    """
    logging.info('@chisel.mteval.assess %s', func.__name__)
    _ASSESS_[func.__name__] = func


def cleanup(func):
    """decorate func with mteval.cleanup in order to prepare your mteval to deal with a new source
    func()
    """
    _CLEANUP_.append(func)
    return func


def reset(func):
    """decorate func with mteval.reset in order to prepare your mteval to deal with a new training set
    func()"""
    _RESET_.append(func)
    return func


# the following is not meant to be used as decorators


def load_metrics(metrics):
    for mdef in metrics:
        if os.path.isfile(mdef):
            try:
                logging.info('Loading mteval definitions from file %s', mdef)
                prefix = os.path.dirname(mdef)
                sys.path.append(prefix)
                __import__(os.path.basename(mdef).replace('.py', ''))
                sys.path.remove(prefix)
            except:
                logging.error('Could not load feature definitions from file %s', mdef)
        else:
            try:
                logging.info('Loading mteval definitions from module %s', mdef)
                importlib.import_module(mdef)
            except:
                logging.error('Could not load mteval defitions from module %s', mdef)


def configure_metrics(config):
    """configure mteval modules"""
    [func(config) for func in _CONFIGURE_]


def prepare_training(source, references, hypotheses, consensus=False):
    [func(source, references, hypotheses, consensus) for func in _TRAINING_]


def prepare_decoding(source, evidence, hypotheses, consensus=False):
    [func(source, evidence, hypotheses, consensus) for func in _DECODING_]


def comparisons(c, r):
    return [(fname, func(c, r)) for func, fname in _COMPARE_.iteritems()]


def comparison(c, r, metric):
    return _COMPARE_[metric](c, r)


def assessments(c, r):
    return [(fname, func(c, r)) for func, fname in _ASSESS_.iteritems()]


def assessment(c, r, metric):
    return _ASSESS_[metric](c, r)


def cleanup_metrics():
    [func() for func in _CLEANUP_]


def reset_metrics():
    [func() for func in _RESET_]


def sanity_check(metric_name):
    return metric_name in _COMPARE_ and metric_name in _ASSESS_