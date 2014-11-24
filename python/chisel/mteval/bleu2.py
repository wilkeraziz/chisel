"""
@author waziz
"""
import chisel.mteval as mteval
import logging
from _bleu2 import BLEU, DecodingBLEU, TrainingBLEU


class WrappedBLEU(mteval.LossFunction):

    def __init__(self, alias):
        self.alias_ = alias
        self.bleu_config_ = {}
        self.decoding_bleu_wrapper_ = None
        self.training_bleu_wrapper_ = None

    @property
    def alias(self):
        return self.alias_

    def configure(self, config):
        # copies configuration
        self.bleu_config_ = dict(config)
        # sets default values if necessary
        if 'max_order' not in self.bleu_config_:
            logging.info('BLEU using default max_order=%d', BLEU.DEFAULT_MAX_ORDER)
            self.bleu_config_['max_order'] = BLEU.DEFAULT_MAX_ORDER
        if 'smoothing' not in self.bleu_config_:
            logging.info('BLEU using default smoothing=%s', BLEU.DEFAULT_SMOOTHING)
            self.bleu_config_['smoothing'] = BLEU.DEFAULT_SMOOTHING

    def prepare_decoding(self, src, evidence, hypotheses):
        """
        Compute sufficient statistics for BLEU in decoding mode
        :param src:
        :param EmpiricalDistribution evidence:
        :param EmpiricalDistribution hypotheses:
        """
        assert evidence is hypotheses, 'For now BLEU decoding is supported with Yh == Ye'
        self.decoding_bleu_wrapper_ = DecodingBLEU(evidence, evidence.copy_posterior(), **self.bleu_config_)

    def prepare_training(self, source, references, hypotheses):
        """
        Compute sufficient statistic for BLEU in training mode
        :param source:
        :param references:
        :param EmpiricalDistribution hypotheses:
        :return:
        """
        self.training_bleu_wrapper_ = TrainingBLEU(references, hypotheses, **self.bleu_config_)

    def training_loss(self, c):
        return 1 - self.training_bleu_wrapper_.bleu(c)

    def loss(self, c, r):
        return 1 - self.decoding_bleu_wrapper_.bleu(c, r)

    def coloss(self, c):
        return 1 - self.decoding_bleu_wrapper_.cobleu(c)

    def cleanup(self):
        self.decoding_bleu_wrapper_ = None

    def reset(self):
        pass


def construct(alias):
    return WrappedBLEU(alias)