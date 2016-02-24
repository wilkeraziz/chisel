

class Scorer(object):
    """
    Minimum interface for a scorer.
    """

    def __init__(self, sid, n_components):
        """
        :param sid: scorer identifier
        :param n_components: number of components (use 0 for none, >0 for a fixed number of components, -1 for a variable number of sparse components)
        """
        self._sid = sid
        self._n_components = n_components

    @property
    def sid(self):
        return self._sid

    @property
    def n_components(self):
        return self._n_components

    def configure(self, config):
        raise NotImplementedError()

    def preprocess(self, segment):
        pass

    def suffstats(self, hypothesis):
        pass

    def featurize(self, hypothesis):
        pass

    def cleanup(self):
        pass

    def reset(self):
        pass

        


