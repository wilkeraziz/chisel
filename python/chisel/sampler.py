"""
:Authors: - Wilker Aziz
"""
import collections
import logging
import sys
import argparse
import math
import os
import traceback
import itertools
from tabulate import tabulate
from time import time
from multiprocessing import Pool
from functools import partial
import chisel.ff as ff
import chisel.ffpp.manager as ffpp
import chisel.cdeclib as cdeclib
import numpy as np
from chisel.util import fpairs2str, dict2str, fmap_dot, scaled_fmap
from chisel.util import resample as do_resample
from chisel.util.config import configure
from chisel.util.iotools import SegmentMetaData
from chisel.smt import SVector, Tree, Derivation
from chisel.instrumental import Sampler, KLOptimiser


def argparse_and_config():
    parser = argparse.ArgumentParser(description='MC sampler for hiero models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config',
                        type=str, help='configuration file')
    parser.add_argument('workspace',
                        type=str, default=None,
                        help='samples will be written to $workspace/samples/$i')
    parser.add_argument("--target-scaling",
                        type=float, default=1.0,
                        help="scaling parameter for the target model")
    parser.add_argument("--proxy-scaling",
                        type=float, default=1.0,
                        help="scaling parameter for the proxy model")
    parser.add_argument("--avgcoeff",
                        type=float, default=1.0,
                        help="recency coefficient for the moving average in KL optimisation")
    parser.add_argument("--klreg",
                        type=str, default='none', choices=['none', 'L1', 'L2'],
                        help="regulariser for KL optimisation")
    parser.add_argument("--klregw",
                        type=float, default=0.0,
                        help="regulariser's weight for KL optimisation")
    parser.add_argument("--klnew",
                        type=float, default=0.0,
                        help="weight of an additional regulariser based on the effective sample size")
    parser.add_argument("--samples",
                        type=int, default=100,
                        help="number of samples")
    parser.add_argument("--tuning-samples",
                        type=int, default=100,
                        help="number of samples when tuning the proxy")
    parser.add_argument("--resampling",
                        action='store_true',
                        help="resample the importance weights")
    parser.add_argument("--grammars",
                        type=str,
                        help="where to find grammars (grammar files are expected to be named grammar.$i.sgm, "
                             "with $i 0-based)")
    parser.add_argument('--jobs', '-j',
                        type=int, default=2,
                        help='number of processes')
    parser.add_argument('--sortby',
                        type=str, default='none',
                        choices=['n', 'p', 'q', 'r', 'nr'],
                        help='sort results by a specific column')
    parser.add_argument('--verbose', '-v',
                        action='count',
                        help='increases verbosity')
    parser.add_argument('--tune', '-T',
                        action='store_true',
                        help='Tune the proxy distribution')


    args, config, failed = configure(parser,
                                     set_defaults=['chisel:model', 'chisel:sampler'],
                                     required_sections=['proxy', 'target', 'cdec'],
                                     configure_logging=True)
    logging.debug('arguments: %s', vars(args))

    if failed:
        sys.exit(1)

    return args, config


def sample_and_save(seg, proxy_weights, target_weights, cdec_cfg_str, output_dir, options):
    try:
        sampler = Sampler(seg, proxy_weights, target_weights, cdec_cfg_str)
        if options.tune:  # perhaps we tune Q by optimising KL(q||p)
            optimiser = KLOptimiser(seg, 
                    options.tuning_samples, 
                    proxy_weights, 
                    target_weights, 
                    cdec_cfg_str,
                    regulariser=options.klreg,
                    regulariser_weight=options.klregw,
                    ne_weight=options.klnew,
                    avgcoeff=options.avgcoeff)
            optq = optimiser.optimise()  # optimise the proxy
            sampler.reweight(optq)  # reweight the forest
        # samples
        samples = sampler.sample(options.samples)
        sampler.save(samples, output_dir)
        return seg.id, len(samples), len(set(s.projection for s in samples))
    except:
        raise Exception('job={0} exception={1}'.format(seg.id,
                                                       ''.join(traceback.format_exception(*sys.exc_info()))))


def main():
    options, config = argparse_and_config()

    # make output dir
    output_dir = '{0}/samples'.format(options.workspace)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    logging.info('Writing samples to: %s', output_dir)

    # cdec configuration string
    # TODO: I need to allow duplicates in cdec-features, currently duplicates overwrite each other
    # perhaps the easiest to do is to separate cdec-features from chisel.ini
    cdec_cfg_string = cdeclib.make_cdec_config_string(config.items('cdec'), config.items('cdec:features'))
    logging.debug('cdec.ini: %s', repr(cdec_cfg_string))

    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(config.items('proxy'), options.proxy_scaling)
    logging.debug('proxy (scaling=%f): %s', options.proxy_scaling, dict2str(proxy_weights, sort=True))

    # parameters of the target distribution
    target_weights = scaled_fmap(config.items('target'), options.target_scaling)
    logging.debug('target (scaling=%f): %s', options.target_scaling, dict2str(target_weights, sort=True))

    # loads scorer modules
    if config.has_section('chisel:scorers'):
        ff.load_scorers(config.items('chisel:scorers'))

    # scorers' configuration
    if config.has_section('chisel:scorers:config'):
        scorers_config = dict(config.items('chisel:scorers:config'))
    else:
        scorers_config = {}
    logging.debug('chisel:scorers:config: %s', scorers_config)
    # configure scorers
    ff.configure_scorers(scorers_config)

    
    # FF++: an improved FF framework
    # 1. load implementations
    if config.has_section('chisel:scorers++'):
        scorerspp_map = dict(config.items('chisel:scorers++'))
        ffpp.load_scorers(scorerspp_map.iteritems())
    # 2. config scorers
    if config.has_section('chisel:scorers++:config'):
        scorerspp_config = dict(config.items('chisel:scorers++:config'))
    else:
        scorerspp_config = {}
    logging.info('chisel:scorers++:config: %s', scorerspp_config)
    ffpp.configure_scorers(scorerspp_config)
    # FF++ done
    
    # logs which features were added to the proxy
    extra_features = {k: v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.debug('Extra features: %s', extra_features)

    # reads segments from input
    segments = [SegmentMetaData.parse(line.strip(),
                                      'cdec',
                                      grammar_dir=options.grammars)
                for line in sys.stdin]  

    # sample and save results
    logging.info('Distributing %d segments to %d jobs', len(segments), options.jobs)
    pool = Pool(options.jobs)
    feedback = pool.map(partial(sample_and_save,
                               proxy_weights=proxy_weights,
                               target_weights=target_weights,
                               cdec_cfg_str=cdec_cfg_string,
                               output_dir=output_dir,
                               options=options),
                       segments)
    feedback.sort(key=lambda t: t[0])  # sort by segment id
    
    print tabulate(feedback, headers=('job', 'derivations', 'strings'), tablefmt='pipe')
            

if __name__ == '__main__':
    main()
