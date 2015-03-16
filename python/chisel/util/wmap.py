import numpy as np
import itertools
from collections import defaultdict


class WMap(object):

    def __init__(self, pairs):
        self.features_ = tuple(k for k, v in pairs)
        self.weights_ = np.array([v for k, v in pairs], float)

    def __len__(self):
        return len(self.weights_)

    def iteritems(self):
        return itertools.izip(self.features_, self.weights_)

    @property
    def features(self):
        return self.features_

    @property
    def asarray(self):
        return self.weights_

    def copy_array(self):
        return np.array(self.weights_)

    def asdict(self):
        return defaultdict(None, zip(self.features_, self.weights_))

    def update(self, array):
        assert len(array) == len(self.weights_), 'Wrong dimensionality'
        self.weights_ = array

    def __str__(self):
        return ' '.join('{0}={1}'.format(f, v) for f, v in itertools.izip(self.features_, self.weights_))

    def copy(self):
        return WMap(zip(self.features_, self.weights_))


class JointWMap(object):

    def __init__(self, proxy_wmap, target_wmap):
        self.proxy_ = proxy_wmap
        self.target_ = target_wmap

    def __len__(self):
        return len(self.proxy_) + len(self.target_)

    @property
    def proxy(self):
        return self.proxy_

    @property
    def target(self):
        return self.target_

    def iteritems(self):
        return itertools.chain(self.proxy_.iteritems(), self.target_.iteritems())
    
    @property
    def features(self):
        return self.proxy_.features + self.target_.features

    @property
    def asarray(self):
        return np.concatenate((self.proxy_.asarray, self.target_.asarray))

    def update(self, array):
        self.proxy_.update(array[:len(self.proxy_)])
        self.target_.update(array[len(self.proxy_):])

    def concatenate(self, proxy, target):
        return np.concatenate((proxy, target))

    def __str__(self):
        return '{0} ||| {1}'.format(str(self.proxy_), str(self.target_))

    def copy(self):
        return JointWMap(self.proxy_.copy(), self.target_.copy())


