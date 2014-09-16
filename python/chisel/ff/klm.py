"""
LM features with KenLM.
* these features require configuration (the path to the LM to be loaded)
* we use suffstats to query KenLM for the full scores 
* we use cleanup to clean after suffstats
* we define a few features
    * LanguageModel is the conditional p(w|context)
    * LanguageModel_OOV is the number of OOV words
    * NGramOrder_k where 1 <= k <= max_order is the number of in-vocab k-grams

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
    model = kenlm.LanguageModel(config['KLanguageModel'])

@chisel.ff.suffstats
def full_scores(hypothesis):
    """full_scores(hypothesis) -> compute full scores for each ngram"""
    global suffstats
    suffstats = tuple(model.full_scores(hypothesis.translation_))

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
