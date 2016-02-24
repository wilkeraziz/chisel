import os
import importlib
import sys
import logging

_SCORERS_ = []

class ScorerData(object):

    def __init__(self, sid, impl):
        self.sid = sid
        self.impl = impl
        self.config = None

    def __repr__(self):
        return 'ScorerData(%r, %r)' % (self.sid, self.impl)


def load_scorers(scorers):
    """Loads modules and instantiates scorers."""
    for sid, sdef in scorers:
        module = None
        if os.path.isfile(sdef):
            try:
                logging.info('Loading additional feature definitions from file %s', sdef)
                prefix = os.path.dirname(sdef)
                sys.path.append(prefix)
                module = __import__(os.path.basename(sdef).replace('.py', ''))
                sys.path.remove(prefix)
            except:
                logging.error('Could not load feature definitions from file %s', sdef)
        else:
            try:
                logging.info('Loading additional feature definitions from module %s', sdef)
                module = importlib.import_module(sdef)
            except:
                logging.error('Could not load feature defitions from module %s', sdef)
        if module is None:
            raise NotImplementedError('Could not load module associated with %s: %s' % (fname, sdef))
        _SCORERS_.append(ScorerData(sid, module.get_instance(sid)))


def configure_scorers(config):
    """Configures the scorer objects"""
    for scorer in _SCORERS_:
        scorer.config = {key[len(scorer.sid) + 1:]: value for key, value in config.iteritems() if key.startswith('{0}.'.format(scorer.sid))}
        logging.info('Configuring %s: %s', scorer.sid, scorer.config)
        scorer.impl.configure(scorer.config)

def preprocess_input(segment):
    """preprocess input"""
    [scorer.impl.preprocess(segment) for scorer in _SCORERS_]

def reset_scorers():
    """resets scorers to a null state"""
    [scorer.impl.reset() for scorer in _SCORERS_]

def compute_features(hyp, pairs=None):
    """
    compute_features(hypothesis) -> list of named feature values (i.e. pairs of the kind (fname, fvalue))
    """
    # 1) give scorers the chance to prepare some sufficient statistics
    for scorer in _SCORERS_:
        scorer.impl.suffstats(hyp)

    # 2) evaluate scorers
    if pairs is None:
        pairs = []
    for scorer in _SCORERS_:
        n_comps = scorer.impl.n_components
        if n_comps == 0:
            scorer.impl.featurize(hyp)
        elif n_comps == 1:
            pairs.append(scorer.impl.featurize(hyp))
        else:
            pairs.extend(scorer.impl.featurize(hyp))

    # 3) give ffs the chance to clean up
    [scorer.impl.cleanup() for scorer in _SCORERS_]

    return pairs

