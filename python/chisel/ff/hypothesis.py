__author__ = 'waziz'


class Hypothesis(object):
    """
    Interface exposed to scorers
    """
    def __init__(self, source, translation):
        self.source_ = source
        self.translation_ = translation

    @property
    def src(self):
        return self.source_

    @property
    def tgt(self):
        return self.translation_