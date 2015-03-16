__author__ = 'waziz'

from itertools import izip
import numpy as np


def scaled_fmap(fmap, scaling=1.0):
    """Returns a feature map scaled by a constant"""
    return {k: v*scaling for k, v in fmap.iteritems()}


def fmap_dot(fmap, wmap):
    return sum(fmap.get(fname, 0) * fweight for fname, fweight in wmap.iteritems())


def str2fmap(line):
    """converts a string of the type 'f1=v1 f2=v2' into a feature map {f1: v1, f2: v2}"""
    return {k: float(v) for k, v in (pair.split('=') for pair in line.split())}


def fpairs2str(iterable):
    """converts an iterable of feature-value pairs into string"""
    return ' '.join('%s=%s' % (k, str(v)) for k, v in iterable)


def dict2str(d, separator='=', sort=False, reverse=False):
    """converts an iterable of feature-value pairs into string"""
    if sort:
        return ' '.join('{0}{1}{2}'.format(k, separator, v) for k, v in sorted(d.iteritems(), reverse=reverse))
    else:
        return ' '.join('{0}{1}{2}'.format(k, separator, v) for k, v in d.iteritems())


def npvec2str(nparray, fnames=None):
    """converts an array of feature values into a string (fnames can be provided)"""
    if fnames is None:
        return ' '.join(str(fvalue) for fvalue in nparray)
    else:
        return ' '.join('{0}={1}'.format(fname, fvalue) for fname, fvalue in izip(fnames, nparray))


def kv2str(key, value, named=True):
    return '{0}={1}'.format(key, value) if named else str(value)


def resample(p, size):
    """Resample elements according to a distribution p and returns an empirical distribution"""
    support = p.size
    hist, edges = np.histogram(np.random.choice(np.arange(support), size, p=p), bins=np.arange(support + 1), density=True)
    return hist


def obj2id(element, vocab):
    v = vocab.get(element, None)
    if v is None:
        v = len(vocab)
        vocab[element] = v
    return v
