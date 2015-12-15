__author__ = 'waziz'

import cdec
import logging
from collections import defaultdict
from chisel.util.logtools import debugtime
from chisel.util.iotools import smart_ropen


def make_cdec_config_string(cdec_items, cdec_features_items):
    """
    Create the string equivalent of a cdec.ini
    :param cdec_items: list of pairs in the section [cdec]
    :param cdec_features_items: list of pairs in the section [cdec:features]
    :return str: the configuration string
    """
    return '{0}\n{1}'.format('\n'.join('{0}={1}'.format(k, v) for k, v in cdec_items),
                             '\n'.join('feature_function={0} {1}'.format(k, v) for k, v in cdec_features_items))


def create_decoder_from_files(cdec_ini, weights_file, scaling=1.0):
    """
    Creates an instance of cdec.Decoder
    :param cdec_ini: cdec's configuration file
    :param weights_file: parameters of the instrumental distribution
    :param scaling: scaling factor (defaults to 1.0)
    :return: an instance of cdec.Decoder
    """
    with open(cdec_ini) as f:
        config_str = f.read()
        logging.info('cdec.ini:\n\t%s', '\n\t'.join(config_str.strip().split('\n')))
        # perhaps make sure formalism=scfg and intersection_strategy=full?
        # decoder = cdec.Decoder(config_str=config_str, formalism='scfg', intersection_strategy='Full')
        decoder = cdec.Decoder(config_str=config_str)

    logging.info('Loading weights: %s', weights_file)
    decoder.read_weights(weights_file, scaling)
    # logging.info('Weights: %s', dict(decoder.weights))
    return decoder


def create_decoder(config_str, weights):
    """
    Creates an instance of cdec.Decoder
    :param config_str: cdec's configuration string
    :param dict weights: parameters of the instrumental distribution
    :return: an instance of cdec.Decoder
    """
    # perhaps make sure formalism=scfg and intersection_strategy=full?
    # decoder = cdec.Decoder(config_str=config_str, formalism='scfg', intersection_strategy='Full')
    decoder = cdec.Decoder(config_str=config_str)
    for k, v in weights.iteritems():
        decoder.weights[k] = v

    return decoder


def build_proxy(input_str, grammar_file, decoder):
    """
    :param input_str: actual string to be translated
    :param grammar_file: where the grammar is stored
    :param decoder: an instance of cdec.Decoder (see create_decoder)
    :return: hypergraph
    """
    logging.info('Loading grammar: %s', grammar_file)
    with smart_ropen(grammar_file) as f:
        grammar = f.read()
    logging.info('Composing the forest')
    forest = decoder.translate(input_str, grammar=grammar)
    return forest


@debugtime('cdeclib.sample [time]')
def sample(forest, n):
    """
    Samples n derivations from forest
    :param forest: hypergraph
    :param n: number of samples
    :return: a defaultdict mapping from a translation string to a list of tuples of the kind (fpairs, dot)
    """
    sampledict = defaultdict(list)
    for sample_str, sample_dot, sample_fmap in forest.sample_hypotheses(n):
        sampledict[sample_str.encode('utf8')].append((tuple(sample_fmap), sample_dot))
    return sampledict


def make_sparse_vector(fmap):
    """returns a cdec.SparseVector from a dict-like object"""
    weights = cdec.SparseVector()
    for fname, fval in fmap.iteritems():
        weights[fname] = fval
    return weights


def make_dense_vector(fmap):
    """returns a cdec.DenseVector from a dict-like object"""
    weights = cdec.DenseVector()
    for fname, fval in fmap.iteritems():
        weights[fname] = fval
    return weights
