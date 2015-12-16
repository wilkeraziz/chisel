"""

:Authors: - Wilker Aziz
"""

import argparse
import logging
import sys
from chisel.learning.newdriver import Driver
from chisel.util.config import section_literal_eval, configure


def main(args, config):
    driver = Driver(*argparse_and_config())


def argparse_and_config():
    parser = argparse.ArgumentParser(description='Tuning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', type=str, help="configuration file")
    parser.add_argument("workspace",
                        type=str, default=None,
                        help="where samples can be found and where decisions are placed")
    parser.add_argument("dev", type=str,
                        help="development set")
    parser.add_argument('--dev-alias', type=str, default='dev',
            help='Change the alias of the dev set')
    parser.add_argument('--no-eval-dev', action='store_true', default=False,
            help='Do not assess the dev set at the beginning of an iteration')
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from a certain iteration (requires the config file of the preceding run)")
    parser.add_argument("--metric", type=str, default='bleu',
                        help="similarity function")
    parser.add_argument("--maxiter", '-M', type=int, default=10,
                        help="Maximum number of iterations")
    parser.add_argument('--samples', type=int, default=1000, # nargs='+', default=[1000],
            help='Sampling schedule: number of samples and number of iterations (multiple allowed)')
    # Optimisation
    parser.add_argument("--order", type=str, default='pq', choices=['pq', 'qp'],
            help="Order in which to optimise parameters: p then q, or q then p")
    parser.add_argument("--qopt", type=str, default='minkl', choices=['minkl', 'maxelb', 'minvar'],
                        help="Optimisation method for instrumental distribution")
    # SGD target
    parser.add_argument("--psgd", type=int, nargs=2, default=[10, 20],
                        help="Number of iterations and function evaluations for target optimisation")
    parser.add_argument("--ptol", type=float, nargs=2, default=[1e-4, 1e-4],
                        help="f-tol and g-tol in target optimisation")
    parser.add_argument("--pL2", type=float, default=0.0,
                        help="Weight of L2 regulariser in target optimisation")
    parser.add_argument("--Tp", type=float, default=0.0,
            help="Temperature parameter for target's entropic prior")
    parser.add_argument("--minTp", type=float, default=0.0,
            help="Minimum temperature for target's entropic prior")
    parser.add_argument("--pcooling-factor", type=float, default=1.0,
            help="Cooling factor in target optimisation")
    parser.add_argument("--pcooling-lag", type=int, default=1,
            help="Number of iterations between cooling in target optimisation")
    # SGD proxy
    parser.add_argument("--qsgd", type=int, nargs=2, default=[5, 10],
                        help="Number of iterations and function evaluations for proxy optimisation")
    parser.add_argument("--qtol", type=float, nargs=2, default=[1e-4, 1e-4],
                        help="f-tol and g-tol in proxy optimisation")
    parser.add_argument("--qL2", type=float, default=0.0,
                        help="Weight of L2 regulariser in proxy optimisation")
    parser.add_argument("--Tq", type=float, default=0.0,
            help="Temperature parameter for proxy's entropic prior")
    parser.add_argument("--minTq", type=float, default=0.0,
            help="Minimum temperature for proxy's entropic prior")
    parser.add_argument("--qcooling-factor", type=float, default=1.0,
            help="Cooling factor in proxy optimisation")
    parser.add_argument("--qcooling-lag", type=int, default=1,
            help="Number of iterations between cooling in proxy optimisation")
    # General
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument("--devtest", type=str,
                        help="devtest set")
    parser.add_argument('--devtest-alias', type=str, default='devtest',
            help='Change the alias of the devtest set')
    parser.add_argument("--devtest-grammar", type=str,
                        help="grammars for the devtest set")
    parser.add_argument("--alias", type=str,
                        help="an alias for the experiment")
    parser.add_argument('--default', type=float, default=None,
                        help='initialise all weights with a default value, if not given, we start from the values already specified in the config file')
    parser.add_argument('--consensus',
                        action='store_true',
                        help='consensus training instead of MBR training')
    parser.add_argument('--save-loss',
                        action='store_true',
                        help='Store sample loss at every iteration')
    parser.add_argument('--scoring-tool', type=str,
            default='/Users/waziz/workspace/github/cdec/mteval/fast_score',
            help='a scoring tool such as fast_score')
    parser.add_argument('--verbose', '-v',
                        action='count',
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
    #config.set('chisel:decision', 'nbest', repr(args.nbest))
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
    #config.set('chisel:learning', 'nbest', repr(args.nbest))
    config.set('chisel:learning', 'alias', repr(args.alias))
    config.set('chisel:learning', 'default', repr(args.default))
    config.set('chisel:learning', 'jobs', repr(args.jobs))
    config.set('chisel:learning', 'consensus', repr(args.consensus))
    config.set('chisel:learning', 'psgd', repr(args.psgd))
    config.set('chisel:learning', 'ptol', repr(args.ptol))
    config.set('chisel:learning', 'pL2', repr(args.pL2))
    config.set('chisel:learning', 'Tp', repr(args.Tp))
    config.set('chisel:learning', 'qsgd', repr(args.qsgd))
    config.set('chisel:learning', 'qtol', repr(args.qtol))
    config.set('chisel:learning', 'qL2', repr(args.qL2))
    config.set('chisel:learning', 'Tq', repr(args.Tq))
    config.set('chisel:learning', 'qopt', repr(args.qopt))
    

    return args, config


if __name__ == '__main__':
    main(*argparse_and_config())
