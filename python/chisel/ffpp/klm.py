import chisel
import kenlm
import collections
import sys
from chisel.ffpp.scorer import Scorer


class KenLM(Scorer):

    def __init__(self, sid):
        super(KenLM, self).__init__(sid, 2)
        self._name = sid
        self._model = None
        self._suffstats = None

    def configure(self, config):
        if 'model' not in config:
            raise Exception('Perhaps you forgot to configure {0}.model in your chisel.ini file?'.format(self.sid))
        self._model = kenlm.LanguageModel(config['model'])
        if 'name' in config:
            self._name = config['name']

    def suffstats(self, hypothesis):
        """full_scores(hypothesis) -> compute full scores for each ngram"""
        self._suffstats = tuple(self._model.full_scores(hypothesis.tgt))

    def featurize(self, hypothesis):
        total_prob = 0
        total_oov = 0
        for prob, length, oov in self._suffstats:
            total_prob += prob
            total_oov += oov
        return (self._name, total_prob), ('{0}_OOV'.format(self._name), total_oov)

    def cleanup(self):
        self._suffstats = None


def get_instance(sid):
    return KenLM(sid)

