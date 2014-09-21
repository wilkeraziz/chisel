"""
n-gram LM features with KenLM.

Usage:

* this module requires configuration (via chisel.ini)
    
        KLanguageModel=<path to trained model>

* the following features are defined

    1. LanguageModel is the LM log-probability of the sentence
    2. LanguageModel_OOV is the number of OOV words
    3. NGramOrder_k where 1 <= k <= max_order is the number of in-vocab k-grams

Implementation details:

* @chisel.ff.suffstats is used to query KenLM for the full scores before computing specific features
* @chisel.ff.cleanup is used to clean after suffstats

@author waziz
"""
import chisel
import kenlm
import collections
import sys

model = None
suffstats = None

@chisel.ff.configure
def configure(config):
    global model
    if 'KLanguageModel' not in config:
        raise Exception('Perhaps you forgot to configure `KLanguageModel=<path to LM>` in your chisel.ini file?')
    model = kenlm.LanguageModel(config['KLanguageModel'])

@chisel.ff.suffstats
def full_scores(hypothesis):
    """full_scores(hypothesis) -> compute full scores for each ngram"""
    global suffstats
    suffstats = tuple(model.full_scores(hypothesis.tgt))

@chisel.ff.cleanup
def cleanup():
    global suffstats
    suffstats = None

@chisel.ff.features('LanguageModel', 'LanguageModel_OOV')
def KLanguageModel(hypothesis):
    total_prob = 0
    total_oov = 0
    for prob, length, oov in suffstats:
        total_prob += prob
        total_oov += oov
    return (total_prob, total_oov)

@chisel.ff.sparse
def NGramOrder(hypothesis):
    counter = collections.Counter(length for prob, length, oov in suffstats)
    return counter.iteritems()

if __name__ == '__main__':
    print __doc__
