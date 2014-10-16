__author__ = 'waziz'

from itertools import izip


def str2fmap(line):
    """converts a string of the type 'f1=v1 f2=v2' into a feature map {f1: v1, f2: v2}"""
    return {k: float(v) for k, v in (pair.split('=') for pair in line.split())}


def fpairs2str(iterable):
    """converts an iterable of feature-value pairs into string"""
    return ' '.join('%s=%s' % (k, str(v)) for k, v in iterable)


def npvec2str(nparray, fnames=None):
    """converts an array of feature values into a string (fnames can be provided)"""
    if fnames is None:
        return ' '.join(str(fvalue) for fvalue in nparray)
    else:
        return ' '.join('{0}={1}'.format(fname, fvalue) for fname, fvalue in izip(fnames, nparray))


def kv2str(key, value, named=True):
    return '{0}={1}'.format(key, value) if named else str(value)

