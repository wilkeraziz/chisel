"""
@author waziz
"""
import logging
import sys
import traceback
import os
import argparse
from multiprocessing import Pool
from decoder import MBR, MAP, consensus
from util.io import read_sampled_derivations, read_block, list_numbered_files
from decoder.estimates import EmpiricalDistribution
from smt import groupby, KBestSolution
from functools import partial
from util import scaled_fmap, dict2str
from util.config import section_literal_eval, configure
from util.wmap import WMap
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


def make_decisions(job_desc, headers, options):  #, q_wmap, p_wmap):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    jid, path = job_desc
    try:
        derivations, q_wmap, p_wmap = read_sampled_derivations(open(path, 'r'))
        empdist = EmpiricalDistribution(derivations,
                                        q_wmap=q_wmap,
                                        p_wmap=p_wmap,
                                        get_yield=lambda d: d.tree.projection)

        logging.info('%d derivations and %d unique strings', len(derivations), len(empdist))

        solutions = {}

        if options.map:
            # print 'MAP:'
            posterior = MAP(empdist, normalise=True)
            solutions['MAP'] = pack_nbest(empdist, posterior, options.nbest, reward=True)

        if options.mbr or options.consensus:
            mteval.prepare_decoding(None, empdist, empdist)
            if options.mbr:
                eb_gains = MBR(empdist, options.metric, normalise=True)
                solutions['MBR'] = pack_nbest(empdist, eb_gains, options.nbest, reward=False)

            if options.consensus:
                co_gains = consensus(empdist, options.metric, normalise=True)
                solutions['consensus'] = pack_nbest(empdist, co_gains, options.nbest, reward=False)

        return solutions

    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def create_decision_rule_dir(workspace, decision_rule, metric_name=None):
    if metric_name is None:
        output_dir = '{0}/decisions/{1}'.format(workspace, decision_rule)
    else:
        output_dir = '{0}/decisions/{1}-{2}'.format(workspace, decision_rule, metric_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir


def decide_and_save(job_desc, headers, options, output_dirs):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    jid, path = job_desc
    try:
        # make decisions
        decisions = make_decisions(job_desc, headers, options)  #, q_wmap, p_wmap)
        # write to file if necessary
        for rule, ranking in decisions.iteritems():
            with open('{0}/{1}'.format(output_dirs[rule], jid), 'w') as out:
                print >> out, '\t'.join(['#target', '#p', '#q', '#yield'])
                for solution in ranking:
                    print >> out, solution.format_str(keys=['p', 'q', 'yield'])
                print >> out
        return {rule: solutions[0] for rule, solutions in decisions.iteritems()}
    except:
        raise Exception('job={0} exception={1}'.format(jid, ''.join(traceback.format_exception(*sys.exc_info()))))


def argparse_and_config():
    parser = argparse.ArgumentParser(description='Applies a decision rule to a sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config',
                        type=str,
                        help="configuration file")
    parser.add_argument("workspace",
                        type=str, default=None,
                        help="where samples can be found and where decisions are placed")
#    parser.add_argument("--target-scaling",
#                        type=float, default=1.0,
#                        help="scaling parameter for the target model (default: 1.0)")
#    parser.add_argument("--proxy-scaling",
#                        type=float, default=1.0,
#                        help="scaling parameter for the proxy model (default: 1.0)")
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
#    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')), options.proxy_scaling)
#    proxy_wmap = WMap(proxy_weights.iteritems())
    #logging.debug('proxy (scaling=%f): %s', options.proxy_scaling, dict2str(proxy_weights, sort=True))
#    logging.debug('proxy (scaling=%f): %s', options.proxy_scaling, str(proxy_wmap))

    # parameters of the target distribution
#    target_weights = scaled_fmap(section_literal_eval(config.items('target')), options.target_scaling)
#    target_wmap = WMap(target_weights.iteritems())
#    logging.debug('target (scaling=%f): %s', options.target_scaling, str(target_wmap))

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
    # configure metrics
    mteval.configure(metrics_config)

    # gather decision rules to be run
    decision_rules = []
    if options.map:
        decision_rules.append('MAP')
    if options.mbr:
        decision_rules.append('MBR')
    if options.consensus:
        decision_rules.append('consensus')

    # check for input folder
    samples_dir = '{0}/samples'.format(options.workspace)
    if not os.path.isdir(samples_dir):
        raise Exception('If a workspace is set, samples are expected to be found under $workspace/samples')
    logging.info('Reading samples from %s', samples_dir)
    # create output folders
    if not os.path.isdir('{0}/output'.format(options.workspace)):
        os.makedirs('{0}/output'.format(options.workspace))
    output_dirs = {}
    one_best_files = {}
    # TODO: check whether decisions already exist (and warn the user)
    for rule in decision_rules:
        if rule == 'MAP':
            output_dirs[rule] = create_decision_rule_dir(options.workspace, rule)
            one_best_files[rule] = '{0}/output/{1}'.format(options.workspace, rule)
        else:
            output_dirs[rule] = create_decision_rule_dir(options.workspace, rule, options.metric)
            one_best_files[rule] = '{0}/output/{1}-{2}'.format(options.workspace, rule, options.metric)
        logging.info("Writing '%s' solutions to %s", rule, output_dirs[rule])
        logging.info("Writing 1-best '%s' yields to %s", rule, one_best_files[rule])

    # TODO: generalise this
    headers = {'derivation': 'd', 'vector': 'v', 'count': 'n', 'log_ur': 'log_ur', 'importance': 'importance'}

    # read jobs from workspace
    input_files = list_numbered_files(samples_dir)
    jobs = [(fid, input_file) for fid, input_file in input_files]
    logging.info('%d jobs', len(jobs))

    """
    # sometimes I use this for profiling (gotta write a better switch)
    for job in jobs:
        decide_and_save(job, headers=headers,
                               options=options,
                               fnames=target_features,
                               gnames=proxy_features,
                               output_dirs=output_dirs)

    sys.exit(0)
    """

    # run jobs in parallel
    pool = Pool(options.jobs)
    # run decision rules and save them to files
    results = pool.map(partial(decide_and_save,
                               headers=headers,
                               options=options,
                               #q_wmap=proxy_wmap,
                               #p_wmap=target_wmap,
                               output_dirs=output_dirs),
                       jobs)
    # save the 1-best solution for each decision rule in a separate file
    for rule in decision_rules:
        with open(one_best_files[rule], 'wb') as fout:
            for decisions in results:
                best = decisions[rule]  # instance of KBestSolution
                print >> fout, best.solution.Dy.projection


if __name__ == '__main__':
    #import cProfile
    #cProfile.run('main()')
    main()
