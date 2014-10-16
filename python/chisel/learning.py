__author__ = 'waziz'

import argparse
import logging
import sys
from io_utils import read_weights, read_sampled_derivations
from decoder.estimates import EmpiricalDistribution
from smt import groupby

if __name__ == '__main__':

    # TODO:
    # * Consensus string (DeNero)?

    parser = argparse.ArgumentParser(description='SGD')
    parser.add_argument("--metric", type=str, default='ibm_bleu', help="similarity function, one of {ibm_bleu, bleu_p1, unsmoothed_bleu}")
    parser.add_argument("--proxy", type=str, help="initial parameters of the proxy")
    parser.add_argument("--target", type=str, help="initial parameters of the target")
    parser.add_argument("--scaling", type=float, default=1.0, help="scaling parameter for the model (default: 1.0)")

    options = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # parameters of the instrumental distribution
    proxy_weights = read_weights(options.proxy, options.scaling)
    # parameters of the target distribution
    target_weights = read_weights(options.target, options.scaling)

    # TODO: generalise this
    headers = {'derivation': 'd', 'vector': 'v', 'score': 'p', 'count': 'n', 'importance': 'r'}
    derivations = read_sampled_derivations(sys.stdin, headers)

    E2 = EmpiricalDistribution(groupby(derivations, key=lambda d: d.tree.projection),
                               sorted(target_weights.keys()),
                               sorted(proxy_weights.keys()))

    print E2
