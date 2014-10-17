from ConfigParser import RawConfigParser

__author__ = 'waziz'

import logging
import sys
import os
import argparse
from multiprocessing import Pool
from itertools import izip
from decoder import MBR, consensus, MAP
from metric import BLEU
from util.io import read_weights, read_sampled_derivations, next_block, read_block, list_numbered_files
from smt import groupby, KBestSolution
from decoder.estimates import EmpiricalDistribution
from functools import partial
from util import section_literal_eval, scaled_fmap, dict2str


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
    derivations = read_sampled_derivations(iter(block), headers)
    empdist = EmpiricalDistribution(groupby(derivations, key=lambda d: d.tree.projection),
                                    fnames,
                                    gnames)

    logging.info('%d unique strings', len(empdist))

    bleu = None

    solutions = {}

    if options.map:
        # print 'MAP:'
        posterior = MAP(empdist, normalise=True)
        solutions['MAP'] = pack_nbest(empdist, posterior, options.nbest, reward=True)

    if options.mbr:
        # print 'MBR: IBM-BLEU'
        if bleu is None:
            bleu = BLEU(empdist)
        eb_gains = MBR(empdist, bleu, normalise=True)
        solutions['MBR'] = pack_nbest(empdist, eb_gains, options.nbest, reward=True)

    if options.consensus:
        # print 'Consensus: IBM-BLEU'
        if bleu is None:
            bleu = BLEU(empdist)
        co_gains = consensus(empdist, bleu, normalise=True)
        solutions['consensus'] = pack_nbest(empdist, co_gains, options.nbest, reward=True)

    return solutions


def create_output_dir(workspace, decision_rule):
    output_dir = '{0}/{1}'.format(workspace, decision_rule)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir


def decide_and_save(job_desc, headers, options, fnames, gnames, output_dirs):
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


def argparse_and_config():

    parser = argparse.ArgumentParser(description='Applies a decision rule to a sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', type=str, help="configuration file")
    parser.add_argument("--scaling", type=float, default=1.0, help="scaling parameter for the model (default: 1.0)")
    parser.add_argument("--map", action='store_true', help="MAP decoding")
    parser.add_argument("--mbr", action='store_true', help="MBR decoding")
    parser.add_argument("--consensus", action='store_true', help="Consensus (DeNero et al, 2009)")
    parser.add_argument("--metric", type=str, default='ibm_bleu',
                        choices=['ibm_bleu', 'bleu_p1', 'unsmoothed_bleu'],
                        help="similarity function")
    parser.add_argument("--nbest", type=int, default=1, help="number of solutions")
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument("--workspace", type=str, default=None,
                        help="where samples can be found and where decisions are placed")

    # gather options
    args = parser.parse_args()

    # parse config file
    config = RawConfigParser()
    # this is necessary in order not to lowercase the keys
    config.optionxform = str
    config.read(args.config)
    # some command line options may be overwritten by the section 'chisel:sampler' in the config file
    if config.has_section('chisel:decision'):
        decision_options = section_literal_eval(config.items('chisel:decision'))
        parser.set_defaults(**decision_options)
        # reparse options (with new defaults) TODO: find a better way
        args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    return args, config


def main():
    options, config = argparse_and_config()

    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')), options.scaling)
    gnames = sorted(proxy_weights.iterkeys())
    logging.info('proxy: %s', dict2str(proxy_weights, sort=True))

    # parameters of the target distribution
    target_weights = scaled_fmap(section_literal_eval(config.items('target')), options.scaling)
    fnames = sorted(target_weights.iterkeys())
    logging.info('target: %s', dict2str(target_weights, sort=True))

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
            output_dirs[rule] = create_output_dir(options.workspace, rule)
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

    # single_threaded = False
    #if single_threaded:
    #    for jid, job in jobs:
    #        decisions = make_decisions(block, headers, options, fnames, gnames)
    #        for rule, ranking in decisions.iteritems():
    #            print '[%d] %s' % (jid, rule)
    #            for solution in ranking:
    #                print solution.format_str(keys=['p', 'q', 'n', 'yield'], separator=' ', named=True)
    #    sys.exit(0)

    # run jobs in parallel
    if options.workspace:
        # writing to files
        pool = Pool(options.jobs)
        #job_desc, headers, options, fnames, gnames, output_dirs
        pool.map(partial(decide_and_save,
                         headers=headers,
                         options=options,
                         fnames=fnames,
                         gnames=gnames,
                         output_dirs=output_dirs),
                 jobs)
    else:
        # writing to stdout
        pool = Pool(options.jobs)
        results = pool.map(partial(make_decisions,
                                   headers=headers,
                                   options=options,
                                   fnames=fnames,
                                   gnames=gnames), (job for jid, job in jobs))

        for (j, job), decisions in izip(jobs, results):
            for rule, ranking in decisions.iteritems():
                print '[%d] %s' % (j, rule)
                for solution in ranking:
                    print solution.format_str(keys=['p', 'q', 'n', 'yield'], separator=' ', named=True)


if __name__ == '__main__':
    main()