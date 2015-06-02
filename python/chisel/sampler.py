"""
@author waziz
"""
import collections
import logging
import sys
import argparse
import math
import os
from time import time
from multiprocessing import Pool
from functools import partial
from ConfigParser import RawConfigParser
import ff
import cdeclib
import numpy as np
from util import fpairs2str, dict2str, fmap_dot, scaled_fmap
from util import resample as do_resample
from util.config import configure, section_literal_eval
from util.io import SegmentMetaData
import traceback
import itertools
from smt import SVector, Tree, Derivation
from instrumental import Sampler, KLOptimiser
from tabulate import tabulate


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
                        help="scaling parameter for the target model (default: 1.0)")
    parser.add_argument("--proxy-scaling",
                        type=float, default=1.0,
                        help="scaling parameter for the proxy model (default: 1.0)")
    parser.add_argument("--samples",
                        type=int, default=100,
                        help="number of samples (default: 100)")
    parser.add_argument("--tuning-samples",
                        type=int, default=100,
                        help="number of samples when tuning the proxy (default: 100)")
    parser.add_argument("--resampling",
                        action='store_true',
                        help="resample the importance weights")
    parser.add_argument("--input-format",
                        type=str, default='cdec',
                        choices=['plain', 'cdec'],
                        help="'plain': one input sentence per line and requires --grammars; "
                             "'cdec': one input sentence per line plus a grammar path (sgml-formatted)")
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
                        action='store_true',
                        help='increases verbosity')
    parser.add_argument('--tune', '-T',
                        action='store_true',
                        help='Tune the proxy distribution')

    args, config, failed = configure(parser,
                                     set_defaults=['chisel:model', 'chisel:sampler'],
                                     required_sections=['proxy', 'target', 'cdec'],
                                     configure_logging=True)
    logging.debug('arguments: %s', vars(args))

    # additional sanity checks: input format
    if args.input_format == 'plain' and args.grammars is None:
        logging.error("'--input-format plain' requires '--grammars <path>'")
        failed = True

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
                    avgcoeff=1.0)
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
    cdec_cfg_string = cdeclib.make_cdec_config_string(config.items('cdec'), config.items('cdec:features'))
    logging.debug('cdec.ini: %s', repr(cdec_cfg_string))

    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')), options.proxy_scaling)
    logging.debug('proxy (scaling=%f): %s', options.proxy_scaling, dict2str(proxy_weights, sort=True))

    # parameters of the target distribution
    target_weights = scaled_fmap(section_literal_eval(config.items('target')), options.target_scaling)
    logging.debug('target (scaling=%f): %s', options.target_scaling, dict2str(target_weights, sort=True))

    # loads scorer modules
    if config.has_section('chisel:scorers'):
        scorers_map = section_literal_eval(config.items('chisel:scorers'))
        ff.load_scorers(scorers_map.itervalues())

    # scorers' configuration
    if config.has_section('chisel:scorers:config'):
        scorers_config = section_literal_eval(config.items('chisel:scorers:config'))
    else:
        scorers_config = {}
    logging.info('chisel:scorers:config: %s', scorers_config)

    # logs which features were added to the proxy
    extra_features = {k: v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.debug('Extra features: %s', extra_features)

    # configure scorers
    ff.configure_scorers(scorers_config)

    # reads segments from input
    segments = [SegmentMetaData.parse(line.strip(),
                                      options.input_format,
                                      sid=sid,
                                      grammar_dir=options.grammars)
                for sid, line in enumerate(sys.stdin)]  # easy to check variance (just need to multiply this by a number of trials) 

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
