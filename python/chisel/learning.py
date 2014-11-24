from ConfigParser import RawConfigParser

__author__ = 'waziz'

import argparse
import logging
from util import scaled_fmap, dict2str
from util.io import SegmentMetaData
from util.config import configure, section_literal_eval
import sys
import time


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
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument("--devtest", type=str,
                        help="devtest set")
    parser.add_argument("--alias", type=str,
                        default='SGD-{%Y%m%d-%H%M%S}',  # .format(time.strftime('%Y%m%d-%H%M%S')),
                        help="an alias for the experiment output ($workspace/tuning/$alias)")

    args, config, failed = configure(parser,
                                     set_defaults=['chisel:model', 'chisel:learning'],
                                     required_sections=['proxy', 'target'],
                                     configure_logging=True)
    logging.debug('arguments: %s', vars(args))

    if failed:
        sys.exit(1)

    return args, config


def main():
    options, config = argparse_and_config()

    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')))
    proxy_features = sorted(proxy_weights.iterkeys())
    logging.info('proxy: %s', dict2str(proxy_weights, sort=True))

    # parameters of the target distribution
    target_weights = scaled_fmap(section_literal_eval(config.items('target')))
    target_features = sorted(target_weights.iterkeys())
    logging.info('target: %s', dict2str(target_weights, sort=True))

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