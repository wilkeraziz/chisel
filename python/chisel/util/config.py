"""
@author waziz
"""
from ConfigParser import RawConfigParser
import logging
from ast import literal_eval


def section_literal_eval(items):
    return {k: literal_eval(v) for k, v in items}


def configure(parser, set_defaults=[], required_sections=['proxy', 'target', 'cdec'], configure_logging=True):
    """

    :param argparse.ArgumentParser parser:
    :return:
    """
    args = parser.parse_args()

    if configure_logging:
        if args.verbose:
            if args.verbose > 1:
                logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
            else:
                logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    # parse the config file
    config = RawConfigParser()
    # this is necessary in order not to lowercase the keys
    config.optionxform = str
    config.read(args.config)
    # some command line options may be overwritten by the section 'chisel:sampler' in the config file
    for section in set_defaults:

        if config.has_section(section):
            options = section_literal_eval(config.items(section))
            logging.debug('set_defaults [%s]: %s', section, options)
            parser.set_defaults(**options)
            # reparse options (with new defaults) TODO: find a better way
            args = parser.parse_args()

    # required sections
    failed = False
    for section in required_sections:
        # individual configurations
        if not config.has_section(section):
            logging.error("add a [%s] section to the config file", section)
            failed = True

    return args, config, failed
