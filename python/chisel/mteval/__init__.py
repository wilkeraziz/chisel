"""
To add support to your own mteval metric, you need to wrap it exposing the interface mteval.LossFunction

Your module should also define a constructor function named 'construct' which takes exactly one string parameter:

def construct(alias):
    return MyWrappedMetric(alias)

where alias is the name the user chose to give to your metric in chisel's config file
Example:

[chisel:metrics]
bleu='chisel.mteval.bleu'

Then if you check 'chisel.mteval.bleu' you will find a method

def construct(alias):
    return WrappedBLEU(alias)

when chisel calls this method, alias will contain the string 'bleu' as per the configuration file.
"""


"""
Decoding:

    argmin          sum         L(y', y) p(y)
      y' in Yh        y in Ye


    Yh    HYP (Ye)       REF (Yh)
    1     y1,1...y1,N    y'1
    2     y2,1...y2,N    y'2
    ...   ...            ...
    M     yM,1...yM,N    y'M

Learning

    argmin      sum             sum        L(y', y)p(y)
      t,l        (x',y') in T     y in Ye

    T    HYP (Ye)        REF (T)
    1    y1,1...y1,N     y'1,1 y'1,2 ... y'1,r
    2    y2,1...y2,N     y'2,1 y'2,2 ... y'2,r
    ...  ...             ...
    M    yM,1...yM,N     y'M,1 y'M,2 ... y'1,r
"""

import logging
import os
import sys
import importlib


EXPECTED = None
_OBJECTS_ = {}


class LossFunction(object):

    def __init__(self, alias):
        self.alias_ = alias

    @property
    def alias(self):
        return self.alias_

    def configure(self, config):
        pass

    def prepare_training(self, source, references, hypotheses):
        raise NotImplementedError('This method must be overloaded')

    def prepare_decoding(self, source, evidence, hypotheses):
        raise NotImplementedError('This method must be overloaded')

    def training_loss(self, c):
        """
        Returns the loss incurred in choosing candidate c agains the set of references
        :param int c: candidate
        :return: loss
        """
        raise NotImplementedError('This method must be overloaded')

    def loss(self, c, r):
        """
        Returns the loss incurred in choosing candidate c when r is the reference.
        :param int c: candidate
        :param int r: reference
        :return: loss
        """
        raise NotImplementedError('This method must be overloaded')

    def training_coloss(self):
        """
        Returns the consensus loss, where the candidate is represented by a vector of expected features.
        :return: consensus loss
        """
        raise NotImplementedError('This method must be overloaded')

    def coloss(self, c):
        """
        Returns the consensus loss, where the reference is represented by a vector of expected features.
        :param int c: candidate
        :return: consensus loss
        """
        raise NotImplementedError('This method must be overloaded')

    def cleanup(self):
        pass

    def reset(self):
        pass


def load(metrics, subset=None):
    """
    Load MT evaluation metrics.
    :param dict metrics: alias => path to module
    :param subset: if defined, loads only a subset of the metrics
    """
    global _OBJECTS_
    _OBJECTS_ = {}
    for alias, path in metrics.iteritems():
        if subset is not None and alias not in subset:
            logging.debug('mteval.load was requested to ignore: %s', alias)
            continue
        module = None
        if os.path.isfile(path):
            try:
                logging.info('Importing %s from file %s', alias, path)
                prefix = os.path.dirname(path)
                sys.path.append(prefix)
                module = __import__(os.path.basename(path).replace('.py', ''))
                sys.path.remove(prefix)
            except:
                logging.error('Could not find %s definition in %s', alias, path)
        else:
            try:
                logging.info('Importing %s from module %s', alias, path)
                module = importlib.import_module(path)
            except:
                logging.error('Could not find %s defition in %s', alias, path)
        if module is not None:
            _OBJECTS_[alias] = module.construct(alias)


def config_view(config, alias):
    """returns a view of the config dict contain only the keys associated with a certain metric"""
    return {k[len(alias) + 1:]: v for k, v in config.iteritems() if k.startswith('{0}.'.format(alias))}


def configure(config):
    """configure mteval modules"""
    for alias, metric in _OBJECTS_.iteritems():
        view = config_view(config, alias)
        logging.info('Configuring %s: %s', alias, view)
        metric.configure(view)


def prepare_training(source, references, hypotheses):
    [metric.prepare_training(source, references, hypotheses) for metric in _OBJECTS_.itervalues()]


def prepare_decoding(source, evidence, hypotheses):
    [metric.prepare_decoding(source, evidence, hypotheses) for metric in _OBJECTS_.itervalues()]


def loss(c, r, metric):
    return _OBJECTS_[metric].loss(c, r)


def training_loss(c, metric):
    return _OBJECTS_[metric].training_loss(c)


def coloss(c, metric):
    return _OBJECTS_[metric].coloss(c)


def cleanup():
    [metric.cleanup() for metric in _OBJECTS_.itervalues()]


def reset():
    [metric.reset() for metric in _OBJECTS_.itervalues()]


def sanity_check(metric):
    return metric in _OBJECTS_
