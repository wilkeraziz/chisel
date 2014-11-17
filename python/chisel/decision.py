"""
@author waziz
"""

from ConfigParser import RawConfigParser
import logging
import sys
import traceback
import os
import argparse
from multiprocessing import Pool
from itertools import izip
from decoder import MBR, MAP, consensus
from util.io import read_sampled_derivations, next_block, read_block, list_numbered_files
from smt import groupby, KBestSolution
from decoder.estimates import EmpiricalDistribution
from functools import partial
from util import scaled_fmap, dict2str
from util.config import section_literal_eval, configure
import mteval


def sort_by(empdist, scores, reward=True):
    """
    Sort the support of an empirical distribution according to a target score.
    :param empdist: empirical distribution
    :param scores: target scores
    :param reward: polatiry of the scoring function (reward vs penalty)
    :return: a sorted sequence k_0^(N-1) where N is the size of the support and each k_i is an element of the support
    """
    return sorted(range(len(empdist)), key=lambda i: scores[i], reverse=reward)


def pack_nbest(empdist, target, nbest=-1, reward=True):
    """
    Pack the nbest solutions in a list.
    :param empdist: empirical distribution
    :param target:
    :param nbest:
    :param reward:
    :return:
    """
    ranked = sort_by(empdist, target, reward)
    if 0 < nbest < len(ranked):
        ranked = ranked[:nbest]

    return tuple(KBestSolution(k=k,
                               target=target[i],
                               solution=empdist.solution(i)) for k, i in enumerate(ranked))


def make_decisions(block, headers, options, fnames, gnames):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    try:
        derivations = read_sampled_derivations(iter(block), headers)
        empdist = EmpiricalDistribution(groupby(derivations, key=lambda d: d.tree.projection),
                                        fnames,
                                        gnames)

        logging.info('%d unique strings', len(empdist))

        solutions = {}

        if options.map:
            # print 'MAP:'
            posterior = MAP(empdist, normalise=True)
            solutions['MAP'] = pack_nbest(empdist, posterior, options.nbest, reward=True)

        if options.mbr or options.consensus:
            mteval.prepare_decoding(None, empdist, empdist)

            if options.mbr:
                eb_gains = MBR(empdist, options.metric, normalise=True)
                solutions['MBR'] = pack_nbest(empdist, eb_gains, options.nbest, reward=True)

            if options.consensus:
                co_gains = consensus(empdist, options.metric, normalise=True)
                solutions['consensus'] = pack_nbest(empdist, co_gains, options.nbest, reward=True)

        return solutions
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def create_output_dir(workspace, decision_rule, metric_name=None):
    if metric_name is None:
        output_dir = '{0}/{1}'.format(workspace, decision_rule)
    else:
        output_dir = '{0}/{1}-{2}'.format(workspace, decision_rule, metric_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir


def decide_and_save(job_desc, headers, options, fnames, gnames, output_dirs):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    try:
        jid, block = job_desc
        # make decisions
        decisions = make_decisions(block, headers, options, fnames, gnames)
        # write to file if necessary
        for rule, ranking in decisions.iteritems():
            with open('{0}/{1}'.format(output_dirs[rule], jid), 'w') as out:
                print >> out, '\t'.join(['#target', '#p', '#q', '#n', '#yield', '#f', '#g'])
                for solution in ranking:
                    print >> out, solution.format_str(keys=['p', 'q', 'n', 'yield', 'f', 'g'],
                                                      separator='\t', named=False,
                                                      fnames=fnames, gnames=gnames)
                print >> out
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def argparse_and_config():

    parser = argparse.ArgumentParser(description='Applies a decision rule to a sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config',
                        type=str,
                        help="configuration file")
    parser.add_argument("--target-scaling",
                        type=float, default=1.0,
                        help="scaling parameter for the target model (default: 1.0)")
    parser.add_argument("--proxy-scaling",
                        type=float, default=1.0,
                        help="scaling parameter for the proxy model (default: 1.0)")
    parser.add_argument("--map",
                        action='store_true',
                        help="MAP decoding")
    parser.add_argument("--mbr",
                        action='store_true',
                        help="MBR decoding (Kumar and Byrne, 2003)")
    parser.add_argument("--consensus",
                        action='store_true',
                        help="Consensus (DeNero et al, 2009)")
    parser.add_argument("--metric", '-m',
                        type=str, default='bleu',
                        help="similarity function")
    parser.add_argument("--nbest", '-k',
                        type=int, default=1,
                        help="number of solutions")
    parser.add_argument("--jobs", '-j',
                        type=int, default=2,
                        help="number of processes")
    parser.add_argument("--workspace", '-w',
                        type=str, default=None,
                        help="where samples can be found and where decisions are placed")
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='increases verbosity')

    args, config, failed = configure(parser,
                                     set_defaults=['chisel:model', 'chisel:decision'],
                                     required_sections=['proxy', 'target'],
                                     configure_logging=True)
    logging.debug('arguments: %s', vars(args))

    if failed:
        sys.exit(1)

    return args, config


def main():
    options, config = argparse_and_config()

    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')), options.proxy_scaling)
    proxy_features = sorted(proxy_weights.iterkeys())
    logging.debug('proxy (scaling=%f): %s', options.proxy_scaling, dict2str(proxy_weights, sort=True))

    # parameters of the target distribution
    target_weights = scaled_fmap(section_literal_eval(config.items('target')), options.target_scaling)
    target_features = sorted(target_weights.iterkeys())
    logging.debug('target (scaling=%f): %s', options.target_scaling, dict2str(target_weights, sort=True))

    # loads mteval modules
    if config.has_section('chisel:metrics'):
        metrics_map = section_literal_eval(config.items('chisel:metrics'))
        mteval.load_metrics(metrics_map.itervalues())
    else:
        logging.info('Loading BLEU by default')
        mteval.load_metrics(['chisel.mteval.bleu'])

    # metrics' configuration
    if config.has_section('chisel:metrics:config'):
        metrics_config = section_literal_eval(config.items('chisel:metrics:config'))
    else:
        metrics_config = {}
    logging.info('chisel:metrics:config: %s', metrics_config)

    if not mteval.sanity_check(options.metric):
        raise Exception("Perhaps you forgot to include the metric '%s' in the configuration file?" % options.metric)

    # configure metrics
    mteval.configure_metrics(metrics_config)

    # gather decision rules to be run
    decision_rules = []
    if options.map:
        decision_rules.append('MAP')
    if options.mbr:
        decision_rules.append('MBR')
    if options.consensus:
        decision_rules.append('consensus')

    samples_dir = None
    output_dirs = {}

    # if a workspace has been set
    if options.workspace:
        # check for input folder
        samples_dir = '{0}/samples'.format(options.workspace)
        if not os.path.isdir(samples_dir):
            raise Exception('If a workspace is set, samples are expected to be found under $workspace/samples')
        logging.info('Reading samples from %s', samples_dir)
        # create output folders
        # TODO: check whether decisions already exist (and warn the user)
        for rule in decision_rules:
            if rule == 'MAP':
                output_dirs[rule] = create_output_dir(options.workspace, rule)
                logging.info("Writing '%s' decisions to %s", rule, output_dirs[rule])
            else:
                output_dirs[rule] = create_output_dir(options.workspace, rule, options.metric)
                logging.info("Writing '%s' decisions to %s", rule, output_dirs[rule])

    # TODO: generalise this
    headers = {'derivation': 'd', 'vector': 'v', 'score': 'p', 'count': 'n', 'importance': 'r'}

    # read jobs from stdin
    if samples_dir is None:
        jobs = [(bid, block) for bid, block in enumerate(next_block(sys.stdin))]
        logging.info('%d jobs', len(jobs))
    else:
        input_files = list_numbered_files(samples_dir)
        jobs = [(fid, read_block(open(input_file, 'r'))) for fid, input_file in input_files]
        logging.info('%d jobs', len(jobs))

    """
    single_threaded = True
    if single_threaded:
        for jid, job in jobs:
            decisions = make_decisions(job, headers, options, fnames, gnames)
            for rule, ranking in decisions.iteritems():
                print '[%d] %s' % (jid, rule)
                for solution in ranking:
                    print solution.format_str(keys=['p', 'q', 'n', 'yield'], separator=' ', named=True)
        sys.exit(0)
    """

    # run jobs in parallel
    if options.workspace:
        # writing to files
        pool = Pool(options.jobs)
        #job_desc, headers, options, fnames, gnames, output_dirs
        pool.map(partial(decide_and_save,
                         headers=headers,
                         options=options,
                         fnames=target_features,
                         gnames=proxy_features,
                         output_dirs=output_dirs),
                 jobs)
    else:
        # writing to stdout
        pool = Pool(options.jobs)
        results = pool.map(partial(make_decisions,
                                   headers=headers,
                                   options=options,
                                   fnames=target_features,
                                   gnames=proxy_features), (job for jid, job in jobs))
        for (j, job), decisions in izip(jobs, results):
            for rule, ranking in decisions.iteritems():
                print '[%d] %s' % (j, rule)
                for solution in ranking:
                    print solution.format_str(keys=['p', 'q', 'n', 'yield'], separator=' ', named=True)


if __name__ == '__main__':
    main()