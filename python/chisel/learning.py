from ConfigParser import RawConfigParser

__author__ = 'waziz'

import argparse
import logging
from util import section_literal_eval, scaled_fmap, dict2str
from util.io import SegmentMetaData


#def objective(weights, training_data):



def argparse_and_config():
    parser = argparse.ArgumentParser(description='Stochastic Gradient Descent',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', type=str, help="configuration file")
    parser.add_argument("dev", type=str,
                        help="development set")


    parser.add_argument("--metric", type=str, default='bleu',
                        help="similarity function")
    parser.add_argument("--scaling", type=float, default=1.0, help="scaling parameter for the model (default: 1.0)")
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument("--workspace", type=str, default=None,
                        help="where samples can be found and where decisions are placed")
    parser.add_argument("--devtest", type=str,
                        help="devtest set")

    # gather options
    args = parser.parse_args()

    # parse config file
    config = RawConfigParser()
    # this is necessary in order not to lowercase the keys
    config.optionxform = str
    config.read(args.config)
    # some command line options may be overwritten by the section 'chisel:sampler' in the config file
    if config.has_section('chisel:learning'):
        learning_options = section_literal_eval(config.items('chisel:learning'))
        parser.set_defaults(**learning_options)
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

    # TODO: generalise this
    #headers = {'derivation': 'yield', 'vector': 'v', 'score': 'p', 'count': 'n', 'importance': 'r'}
    #derivations = read_sampled_derivations(sys.stdin, headers)

    #empdist = EmpiricalDistribution(groupby(derivations, key=lambda d: d.tree.projection),
    #                                sorted(target_weights.keys()),
    #                                sorted(proxy_weights.keys()))

    #print empdist
    logging.info('Reding dev set: %s', options.dev)
    with open(options.dev, 'r') as f:
        segments = [SegmentMetaData.parse(line.strip(),
                                          'cdec',
                                          sid=sid)
                    for sid, line in enumerate(f)]

        for seg in segments:
            print seg.to_sgm()
            print seg.refs


if __name__ == '__main__':
    main()