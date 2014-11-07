"""
@author waziz
"""

import chisel.mteval as mteval
import logging
from _bleu import BLEU

# dictionary that stores BLEU's configuration (e.g. max_order, smoothing)
bleu_config = None
# bleu wrapper
decoding_bleu_wrapper = None

@mteval.configure
def configure(config):
    global bleu_config
    bleu_config = {}
    if 'bleu.max_order' not in config:
        logging.info('BLEU using default max_order=%d', BLEU.DEFAULT_MAX_ORDER)
        bleu_config['max_order'] = BLEU.DEFAULT_MAX_ORDER
    else:
        bleu_config['max_order'] = config['bleu.max_order']
    if 'bleu.smoothing' not in config:
        logging.info('BLEU using default smoothing=%s', BLEU.DEFAULT_SMOOTHING)
        bleu_config['smoothing'] = BLEU.DEFAULT_SMOOTHING
    else:
        bleu_config['smoothing'] = config['bleu.smoothing']


@mteval.decoding
def suffstats(src, evidence, hypotheses, consensus=False):
    """
    Compute sufficient statistics for BLEU
    :param src:
    :param EmpiricalDistribution evidence:
    :param EmpiricalDistribution hypotheses:
    """
    assert evidence is hypotheses, 'For now BLEU decoding is supported with Yh == Ye'
    global decoding_bleu_wrapper
    decoding_bleu_wrapper = BLEU(evidence, **bleu_config)

@mteval.compare
def bleu(c, r):
    """
    Compare candidate c to reference r
    :param int c:
    :param int r:
    :return: bleu score
    """
    if r is mteval.EXPECTED:
        return decoding_bleu_wrapper.cobleu(c)
    else:
        return decoding_bleu_wrapper.bleu(c, r)


@mteval.assess
def bleu(c):
    """
    :type c: int
    """
    raise NotImplementedError('BLEU training is not yet supported')


@mteval.cleanup
def cleanup():
    global decoding_bleu_wrapper
    decoding_bleu_wrapper = None


@mteval.reset
def reset():
    pass
