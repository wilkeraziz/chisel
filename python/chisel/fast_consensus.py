"""
@author waziz
"""
import logging
import sys
import traceback
import os
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from chisel.decoder import MBR, MAP, consensus
from chisel.util.iotools import read_sampled_derivations, read_block, list_numbered_files
from chisel.decoder.estimates import EmpiricalDistribution
from chisel.smt import groupby, KBestSolution
from chisel.util import scaled_fmap, dict2str
from chisel.util.config import section_literal_eval, configure
from chisel.util.wmap import WMap
from chisel.util.iotools import smart_ropen, smart_wopen
from chisel.learning.newestimates import py
from chisel.mteval.fast_bleu import DecodingBLEU
import chisel.mteval as mteval

def cmpYLPQ(lhs, rhs):
    if lhs[1] != rhs[1]:  # loss
        return cmp(lhs[1], rhs[1])
    elif lhs[2] != rhs[2]:  # posterior
        return cmp(rhs[2], lhs[2])
    elif lhs[3] != rhs[3]:  # proxy
        return cmp(rhs[3], lhs[3])
    else:  # yield
        return cmp(lhs[0], rhs[0])


def create_decision_rule_dir(workspace, decision_rule, metric_name=None):
    if metric_name is None:
        output_dir = '{0}/decisions/{1}'.format(workspace, decision_rule)
    else:
        output_dir = '{0}/decisions/{1}-{2}'.format(workspace, decision_rule, metric_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir


def make_decisions(job_desc, headers, options): 
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    jid, path = job_desc
    try:
        derivations, q_wmap, p_wmap = read_sampled_derivations(smart_ropen(path))
        logging.debug('job=%d derivations=%d empdist...', jid, len(derivations))
        support, posterior, proxy = py(derivations, q_wmap, p_wmap, 
                get_yield=lambda d: d.tree.projection, 
                empirical_q=True, 
                alpha=1.0, beta=1.0)  # TODO: make option
        logging.info('job=%d derivations=%d strings=%d', jid, len(derivations), len(support))

        logging.debug('job=%s consensus...', jid)
        scorer = DecodingBLEU([Dy.leaves for Dy in support], posterior)
        losses = np.array([scorer.loss(Dy.leaves) for Dy in support], float)
        return sorted(zip((Dy.projection for Dy in support), losses, posterior, proxy), cmp=cmpYLPQ)
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def decide_and_save(job_desc, headers, options, output_dir):
    # this code runs in a Pool, thus we wrap in try/except in order to have more informative exceptions
    jid, path = job_desc
    try:
        # make decisions
        ranking = make_decisions(job_desc, headers, options)
        # write to file if necessary
        with smart_wopen('{0}/{1}.gz'.format(output_dir, jid)) as out:  # TODO: save nbest            
            out.write('{0}\n'.format('\t'.join(['#target', '#p', '#q', '#yield'])))
            if options.nbest > 0:
                for y, l, p, q in ranking[0:options.nbest]:
                    out.write('{0}\n'.format('\t'.join(str(x) for x in [l, p, q, y])))
            else:
                for y, l, p, q in ranking:
                    out.write('{0}\n'.format('\t'.join(str(x) for x in [l, p, q, y])))
        return ranking[0]
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

    # check for input folder
    samples_dir = '{0}/samples'.format(options.workspace)
    if not os.path.isdir(samples_dir):
        raise Exception('If a workspace is set, samples are expected to be found under $workspace/samples')
    logging.info('Reading samples from %s', samples_dir)
    # create output folders
    if not os.path.isdir('{0}/output'.format(options.workspace)):
        os.makedirs('{0}/output'.format(options.workspace))
    
    output_dir = create_decision_rule_dir(options.workspace, 'consensus', 'bleu')
    one_best_file = '{0}/output/{1}-{2}'.format(options.workspace, 'consensus', 'bleu')
    logging.info("Writing '%s' solutions to %s", 'consensus', output_dir)
    logging.info("Writing 1-best '%s' yields to %s", 'consensus', one_best_file)

    # TODO: generalise this
    headers = {'derivation': 'd', 'vector': 'v', 'count': 'n', 'log_ur': 'log_ur', 'importance': 'importance'}

    # read jobs from workspace
    input_files = list_numbered_files(samples_dir)
    jobs = [(fid, input_file) for fid, input_file in input_files]
    logging.info('%d jobs', len(jobs))

    # run jobs in parallel
    pool = Pool(options.jobs)
    # run decision rules and save them to files
    results = pool.map(partial(decide_and_save,
                               headers=headers,
                               options=options,
                               output_dir=output_dir),
                       jobs)
    # save the 1-best solution for each decision rule in a separate file
    with smart_wopen(one_best_file) as fout:
        for y, l, p, q in results:
            fout.write('{0}\n'.format(y))


if __name__ == '__main__':
    main()

