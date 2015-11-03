"""

:Authors: - Wilker Aziz
"""

import argparse
import logging
import sys
from learning.newdriver import Driver
from util.config import section_literal_eval, configure


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
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first n iterations")
    parser.add_argument("--maxiter", '-M', type=int, default=10,
                        help="Maximum number of iterations")
    parser.add_argument("--metric", type=str, default='bleu',
                        help="similarity function")
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of samples')
    parser.add_argument('--nbest', type=int, default=-1,
                        help='this has no practical use other than perhaps debugging')
    parser.add_argument("--riskreg", type=float, default=0.0,
                        help="Risk regulariser")
    parser.add_argument("--klreg", type=float, default=0.0,
                        help="KL regulariser")
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument("--devtest", type=str,
                        help="devtest set")
    parser.add_argument("--devtest-grammar", type=str,
                        help="grammars for the devtest set")
    parser.add_argument("--alias", type=str,
                        help="an alias for the experiment")
    parser.add_argument('--default', type=float, 
                        help='initialise all weights with a default value, if not given, we start from the values already specified in the config file')
    parser.add_argument('--consensus',
                        action='store_true',
                        help='consensus training instead of MBR training')
    parser.add_argument('--scoring-tool', type=str,
            default='/Users/waziz/workspace/github/cdec/mteval/fast_score',
            help='a scoring tool such as fast_score')
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


if __name__ == '__main__':
    main(*argparse_and_config())
