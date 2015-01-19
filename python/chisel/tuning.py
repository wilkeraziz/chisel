"""

@author waziz
"""

import argparse
import logging
from util import scaled_fmap, dict2str
from util.io import SegmentMetaData
from util.config import configure, section_literal_eval
import numpy as np
import itertools
import sys
import os
from time import time, strftime
import shlex
from ConfigParser import RawConfigParser
import mteval
import subprocess as sp
import traceback
from multiprocessing import Pool
from decoder.mbr import MBR_training
from decoder.estimates import EmpiricalDistribution
from util.io import read_sampled_derivations, read_block, list_numbered_files
from functools import partial
from smt import groupby


class WMap(object):

    def __init__(self, pairs):
        self.features_ = tuple(k for k, v in pairs)
        self.weights_ = np.array([v for k, v in pairs], float)

    def __len__(self):
        return len(self.weights_)

    def iteritems(self):
        return itertools.izip(self.features_, self.weights_)

    @property
    def features(self):
        return self.features_

    @property
    def asarray(self):
        return self.weights_

    def update(self, array):
        assert len(array) == len(self.weights_), 'Wrong dimensionality'
        self.weights_ = array

    def __str__(self):
        return ' '.join('{0}={1}'.format(f, v) for f, v in itertools.izip(self.features_, self.weights_))


class JointWMap(object):

    def __init__(self, proxy_wmap, target_wmap):
        self.proxy_ = proxy_wmap
        self.target_ = target_wmap

    def __len__(self):
        return len(self.proxy_) + len(self.target_)

    @property
    def proxy(self):
        return self.proxy_

    @property
    def target(self):
        return self.target_

    def iteritems(self):
        return itertools.chain(self.proxy_.iteritems(), self.target_.iteritems())
    
    @property
    def features(self):
        return self.proxy_.features + self.target_.features

    @property
    def asarray(self):
        return np.concatenate((self.proxy_.asarray, self.target_.asarray))

    def update(self, array):
        self.proxy_.update(array[:len(self.proxy_)])
        self.target_.update(array[len(self.proxy_):])

    def __str__(self):
        return '{0} ||| {1}'.format(str(self.proxy_), str(self.target_))
    

def argparse_and_config():
    parser = argparse.ArgumentParser(description='Tuning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', type=str, help="configuration file")
    parser.add_argument("workspace",
                        type=str, default=None,
                        help="where samples can be found and where decisions are placed")
    parser.add_argument("dev", type=str,
                        help="development set")
    parser.add_argument("--metric", type=str, default='bleu',
                        help="similarity function")
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of samples')
    parser.add_argument('--nbest', type=int, default=-1,
                        help='this has no practical use other than perhaps debugging')
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument("--devtest", type=str,
                        help="devtest set")
    parser.add_argument("--alias", type=str,
                        help="an alias for the experiment")
    parser.add_argument('--default', type=float,
                        help='initialise all weights with a default value, if not given, we start from the values already specified in the config file')
    parser.add_argument('--consensus',
                        action='store_true',
                        help='consensus training instead of MBR training')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='increase the verbosity level')

    args, config, failed = configure(parser,
                                     set_defaults=['chisel:model', 'chisel:learning'],
                                     required_sections=['proxy', 'target'],
                                     configure_logging=True)
    logging.debug('arguments: %s', vars(args))

    if failed:
        sys.exit(1)
   
    # reconfig based on command line overrides
    # 1) samples
    config.set('chisel:sampler', 'samples', repr(args.samples))
    config.set('chisel:sampler', 'jobs', repr(args.jobs))
    # 2) decision
    config.set('chisel:decision', 'metric', repr(args.metric))
    config.set('chisel:decision', 'jobs', repr(args.jobs))
    config.set('chisel:decision', 'nbest', repr(args.nbest))
    # 3) metrics
    if config.has_section('chisel:metrics'):
        if args.metric != 'bleu' and not config.has_option('chisel:metrics', 'bleu'):
            raise Exception("Perhaps you forgot to include the metric '%s' in the configuration file?" % args.metric)
        elif args.metric == 'bleu':
            config.set('chisel:metrics', 'bleu', repr('chisel.mteval.bleu'))
    else:
        config.add_section('chisel:metrics')
        config.set('chisel:metrics', 'bleu', repr('chisel.mteval.bleu'))
    # 4) learning
    if not config.has_section('chisel:learning'):
        config.add_section('chisel:learning')
    config.set('chisel:learning', 'metric', repr(args.metric))
    config.set('chisel:learning', 'samples', repr(args.samples))
    config.set('chisel:learning', 'nbest', repr(args.nbest))
    config.set('chisel:learning', 'alias', repr(args.alias))
    config.set('chisel:learning', 'default', repr(args.default))
    config.set('chisel:learning', 'jobs', repr(args.jobs))
    config.set('chisel:learning', 'consensus', repr(args.consensus))
    

    return args, config


def get_joint_wmap(options, config, default=None):
    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')))
    if default is not None:
        proxy_weights = {k: default for k, v in proxy_weights.iteritems()}

    # parameters of the target distribution
    target_weights = scaled_fmap(section_literal_eval(config.items('target')))
    if default is not None:
        target_weights = {k: default for k, v in target_weights.iteritems()}

    return JointWMap(WMap(sorted(proxy_weights.iteritems(), key=lambda (k, v): k)),
        WMap(sorted(target_weights.iteritems(), key=lambda (k, v): k)))


def load_metric(options, config):
    # loads mteval modules
    if config.has_section('chisel:metrics'):
        metrics_map = section_literal_eval(config.items('chisel:metrics'))
    else:
        metrics_map = {'bleu': 'chisel.mteval.bleu'}
    mteval.load(metrics_map, frozenset([options.metric]))

    if not mteval.sanity_check(options.metric):
        raise Exception("Perhaps you forgot to include the metric '%s' in the configuration file?" % options.metric)

    # configure mteval metrics
    if config.has_section('chisel:metrics:config'):
        metrics_config = section_literal_eval(config.items('chisel:metrics:config'))
    else:
        metrics_config = {}
    logging.debug('chisel:metrics:config: %s', metrics_config)
    # mteval.configure(metrics_config)
    return True


def make_workspace(workspace, algorithm, metric, alias=None):
    if alias is None:
        alias = strftime('%Y%m%d-%H%M%S')
    path = '{0}/tuning/{1}-{2}-{3}'.format(workspace, algorithm, metric, alias)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Directory already exists: %s', path)
    os.makedirs('{0}/log'.format(path))
    return path


def prepare_devset(tuning_dir, segments):
    """outputs a dev.input and dev.refs file under the tuning directory"""
    with open('{0}/dev.input'.format(tuning_dir), 'wb') as fi:
        with open('{0}/dev.refs'.format(tuning_dir), 'wb') as fr:
            for seg in segments:
                print >> fi, seg.to_sgm(dump_refs=False)
                print >> fr, ' ||| '.join(str(ref) for ref in seg.refs)


def previous_config(tuning_dir, iteration):
    if iteration == 0:
        return '{0}/base_config.ini'.format(tuning_dir)
    else:
        return '{0}/config{1}.ini'.format(tuning_dir, iteration - 1)


def base_config_file(tuning_dir, args, config, proxy_wmap, target_wmap):

    if config.has_section('proxy'):
        config.remove_section('proxy')
    config.add_section('proxy')
    [config.set('proxy', f, v) for f, v in proxy_wmap.iteritems()]
    
    if config.has_section('target'):
        config.remove_section('target')
    config.add_section('target')
    [config.set('target', f, v) for f, v in target_wmap.iteritems()]
    
    with open('{0}/base_config.ini'.format(tuning_dir), 'wb') as fo:
        config.write(fo)

    return '{0}/base_config.ini'.format(tuning_dir)


def update_config_file(tuning_dir, iteration, proxy_wmap, target_wmap):
    config = RawConfigParser()
    config.optionxform = str

    try:
        config.read(previous_config(tuning_dir, iteration))
    except IOError as e:
        logging.error('[%d] perhaps the previous iteration did not complete successfully', iteration)
        raise e

    [config.set('proxy', f, v) for f, v in proxy_wmap.iteritems()]
    [config.set('target', f, v) for f, v in target_wmap.iteritems()]

    with open('{0}/config{1}.ini'.format(tuning_dir, iteration), 'wb') as fo:
        config.write(fo)

    return '{0}/config{1}.ini'.format(tuning_dir, iteration)


def make_sampling_options(tuning_dir, iteration):
    options = {'config': '{0}/config{1}.ini'.format(tuning_dir, iteration),
            'workspace': '{0}/run{1}'.format(tuning_dir, iteration)}
    cmd_str = 'python -m chisel.sampler %(config)s %(workspace)s' % options
    logging.info('[%d] sampling: %s', iteration, cmd_str)
    cmd_args = shlex.split(cmd_str)
    return cmd_args


def check_samples(tuning_dir, iteration):
    return True


def sample(tuning_dir, iteration, wmap):
    update_config_file(tuning_dir, iteration, wmap.proxy, wmap.target)
    t0 = time()
    with open('{0}/dev.input'.format(tuning_dir), 'rb') as fi:
        with open('{0}/log/sampling.{1}.stdout'.format(tuning_dir, iteration), 'wb') as fo:
            with open('{0}/log/sampling.{1}.stderr'.format(tuning_dir, iteration), 'wb') as fe:
                cmd_args = make_sampling_options(tuning_dir, iteration)
                proc = sp.Popen(cmd_args, stdin=fi, stdout=fo, stderr=fe)
                proc.wait()
    dt = time() - t0
    logging.info('[%d] sampling took %f seconds', iteration, dt)
    if not check_samples(tuning_dir, iteration):
        raise Exception('chisel.sampler appears to have failed at iteration %d', iteration)
    return dt


def make_decision_options(tuning_dir, iteration):
    options = {'config': '{0}/config{1}.ini'.format(tuning_dir, iteration),
            'workspace': '{0}/run{1}'.format(tuning_dir, iteration)}
    cmd_str = 'python -m chisel.decision %(config)s %(workspace)s --consensus' % options
    logging.info('[%d] deciding: %s', iteration, cmd_str)
    cmd_args = shlex.split(cmd_str)
    return cmd_args


def check_decisions(tuning_dir, iteration):
    return True


def decide(tuning_dir, iteration):
    t0 = time()
    with open('{0}/log/decision.{1}.stdout'.format(tuning_dir, iteration), 'wb') as fo:
        with open('{0}/log/decision.{1}.stderr'.format(tuning_dir, iteration), 'wb') as fe:
            cmd_args = make_decision_options(tuning_dir, iteration)
            proc = sp.Popen(cmd_args, stdin=None, stdout=fo, stderr=fe)
            proc.wait()
    dt = time() - t0
    logging.info('[%d] deciding took %f seconds', iteration, dt)
    if not check_decisions(tuning_dir, iteration):
        raise Exception('chisel.decision appears to have failed at iteration %d', iteration)
    return dt


def risk(job, fnames, gnames, options, iteration, headers={'derivation': 'd', 'vector': 'v', 'score': 'p', 'count': 'n', 'importance': 'r'}):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    t0 = time()
    seg, block = job
    try:
        derivations = read_sampled_derivations(iter(block), headers)
        empdist = EmpiricalDistribution(groupby(derivations, key=lambda d: d.tree.projection),
                                        fnames,
                                        gnames)

        logging.info('[%d] (%d) %d unique strings', iteration, seg.id, len(empdist))

        mteval.prepare_training(seg.src, seg.refs, empdist)
        if options.consensus:
            raise Exception('Consensus training is not yet supported.')
        else:
            M = len(empdist)
            losses = np.array([mteval.training_loss(c=h, metric=options.metric) for h, hyp in enumerate(empdist)])
            posterior = empdist.copy_posterior()
            dtheta = np.array([empdist.dtheta(h) for h in range(M)])
            dlambda = np.array([empdist.dlambda(h) for h in range(M)])
            risk = losses.dot(posterior.transpose())
            logging.info('[%d] (%d) risk=%f', iteration, seg.id, risk)

        return losses, posterior, dtheta, dlambda
        #return np.array([losses, posterior, dtheta, dlambda])
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))
   
    dt = time() - t0
    logging.info('[%d] (%d) computing risk took %f seconds', iteration, seg.id, dt)


def risk_pool(tuning_dir, iteration, wmap, devset, options):
    samples_dir = '{0}/run{1}/samples'.format(tuning_dir, iteration)
    if not os.path.isdir(samples_dir):
        raise Exception('[%d] could not find samples' % iteration)
    logging.info('[%d] reading samples from %s', iteration, samples_dir)
    input_files = list_numbered_files(samples_dir)
    jobs = [(devset[fid], read_block(open(input_file, 'r'))) for fid, input_file in input_files]
    logging.info('[%d] %d jobs', iteration, len(jobs))
    # run jobs in parallel
    pool = Pool(options.jobs)
    # run decision rules and save them to files
    results = pool.map(partial(risk,
                               fnames=wmap.target.features,
                               gnames=wmap.proxy.features,
                               options=options,
                               iteration=iteration),
                       jobs)

def main():
    options, config = argparse_and_config()

    # loads and configures the metric we are optimising towards
    load_metric(options, config)

    # model components
    wmap = get_joint_wmap(options, config, default=options.default)
    logging.debug(wmap)

    # create tuning folder
    tuning_dir = make_workspace(options.workspace, 'SGD', options.metric, options.alias)
    logging.info('Working under: %s', tuning_dir)

    # load dev set and separate input and references
    logging.info('Reading dev set: %s', options.dev)
    with open(options.dev, 'r') as f:
        devset = [SegmentMetaData.parse(line.strip(),
                                          'cdec',
                                          sid=sid)
                    for sid, line in enumerate(f)]
    logging.info('%d training instances', len(devset))
    prepare_devset(tuning_dir, devset)

    # prepare config file
    base_config_file(tuning_dir, options, config, wmap.proxy, wmap.target)


    #make_decoding_options(tuning_dir, 0)
    #make_decision_options(tuning_dir, 0)
    iteration = 0
    sample(tuning_dir, iteration, wmap)
    
    # read jobs from workspace
    risk_pool(tuning_dir, iteration, wmap, devset, options)

    # TODO:
    # 1) make a function for the parallel computation of the risk
    # 2) store: loss, posterior, dlambda, dtheta, and risk
    # 3) write the driver: which will control "global" variables such as `tuning dir`, `iteration`, etc.
    # 4) output total empirical risk and total loss at the end of each iteration
    # 5) in the very end, decode devtest, output empirical risk, and total loss


    """
    for x, r in dev:
        D = sample(x, w)
        risk = [L(r, h) * p(h) for h in group_by_yield(D)]

    """




if __name__ == '__main__':
    from learning.driver import Driver
    # main()
    driver = Driver(*argparse_and_config())
