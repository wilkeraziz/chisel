"""
@author waziz
"""
import collections
import logging
import sys
import argparse
import math
import os
from multiprocessing import Pool
from functools import partial
from ConfigParser import RawConfigParser
import ff
import cdeclib
from util import fpairs2str, dict2str, fmap_dot, scaled_fmap
from util.config import configure, section_literal_eval
from util.io import SegmentMetaData
import traceback


class ImportanceSample(object):
    def __init__(self, sample_str, count, fpairs, target_score, log_r):
        """
        :param sample_str: actual sample
        :param count: number of times it was sampled
        :param fpairs: pairs (fname, fvalue)
        :param target_score: ln(up(d)), where up is the unnormalised target distribution
        :param log_r: ln(up(d)) - ln(uq(d)), where uq is the unnormalised instrumental distribution
        """
        self.sample_str_ = sample_str
        self.count_ = count
        self.fpairs_ = fpairs
        self.target_score_ = target_score
        self.log_r_ = log_r

    @property
    def sample_str(self):
        return self.sample_str_

    @property
    def count(self):
        return self.count_

    @property
    def fpairs(self):
        return self.fpairs_

    @property
    def p(self):
        """returns ln(up(d))"""
        return self.target_score_

    @property
    def q(self):
        """returns ln(uq(d))"""
        return self.target_score_ - self.log_r_

    @property
    def r(self):
        """returns ln(up(d)) - ln(uq(d))"""
        return self.log_r_

    def __str__(self):
        return self.format_str()

    def format_str(self, keys='n r p q s v'.split(), separator='\t'):
        """
        Format as string
        :param keys: n (count), r (log importance weight), p (p dot), q (q dot), s (string), d (derivation), v (vector)
        :param separator:
        :return:
        """
        fields = [None] * len(keys)
        for i, k in enumerate(keys):
            if k == 'n':
                x = self.count
            elif k == 'r':
                x = self.r
            elif k == 'p':
                x = self.p
            elif k == 'q':
                x = self.q
            elif k == 's' or k == 'd':  # TODO: return derivation
                x = self.sample_str
            elif k == 'v':
                x = fpairs2str(self.fpairs)
            else:
                raise Exception('Unkonwn field: %s' % k)
            fields[i] = str(x)
        return separator.join(fields)


class Result(object):
    def __init__(self, segment, samples):
        """
        A result associated a segment with its importance samples
        :param segment: the input segment
        :param samples: a list of ImportanceSample objects
        """
        self.segment_ = segment
        self.samples_ = samples

    @property
    def segment(self):
        return self.segment_

    @property
    def samples(self):
        return self.samples_

    def sorted(self, opt):
        if opt == 'n':
            return sorted(self.samples_, key=lambda s: s.count, reverse=True)
        if opt == 'p':
            return sorted(self.samples_, key=lambda s: s.p, reverse=True)
        if opt == 'q':
            return sorted(self.samples_, key=lambda s: s.q, reverse=True)
        if opt == 'r':
            return sorted(self.samples_, key=lambda s: s.r, reverse=True)
        if opt == 'nr':
            return sorted(self.samples_, key=lambda s: s.count * math.exp(s.r), reverse=True)

        return iter(self.samples_)


def sample(segment, n_samples, proxy_weights, target_weights, cdec_config_str='', decoder=None):
    """
    Sample translation derivations for a given segment.
    :param segment: segment to be translated
    :param proxy_weights: parameters of the instrumental distribution
    :param target_weights: parameters of the target distribution
    :param decoder: a possibly null cdec.Decoder object
    :return: an instance of Result
    """
    # creates a decoder if necessary
    if decoder is None:
        decoder = cdeclib.create_decoder(cdec_config_str, proxy_weights)

    logging.info('Translating (%d): %s', segment.id, segment.src)
    # pre-process the input (some scorers might require analysis of the input segment)
    ff.preprocess_input(segment)
    # builds the proxy distribution
    forest = cdeclib.build_proxy(segment.src, segment.grammar, decoder)
    # samples from the proxy distribution
    q_samples = cdeclib.sample(forest, n_samples)
    # header = '\t'.join(['#count', '#translation', '#r', '#qmap', '#qdot', '#pmap', '#pdot'])
    # print header
    # for now we do not have access to alignment
    # ostream = [header]
    is_samples = []
    for sample_str, sample_info in sorted(q_samples.iteritems(), key=lambda pair: len(pair[1]), reverse=True):
        # print >> sys.stderr, len(sample_info), sample_str
        # computes additional features
        extraff = ff.compute_features(ff.Hypothesis(source=segment.src, translation=sample_str))
        # groups vectors associated with equivalent derivations
        counter = collections.Counter(frozenset(fmap.iteritems()) for fmap, _ in sample_info)
        # compute target vectors
        #qdots, pdots = [], []
        for q_fpairs, count in counter.iteritems():
            # start with the features that are used in the proxy
            fmap = dict(q_fpairs)
            # include additional features (must not overwrite proxy features)
            for fname, fvalue in extraff:
                fmap[fname] = fvalue
            # target score (the dot might skip some features, it depends on target_weights)
            pdot = fmap_dot(fmap, target_weights)
            # proxy score
            qdot = fmap_dot(fmap, proxy_weights)
            # make output
            is_samples.append(ImportanceSample(sample_str,
                                               count,
                                               fmap.items(),
                                               pdot,
                                               pdot - qdot))
    # resets scorers to a null state
    ff.reset_scorers()
    return Result(segment, is_samples)


def batch_sample(segments, n_samples, cdec_config_str, proxy_weights, target_weights):
    """
    As :func:`sample`, however processes a batch of segments
    :param segments: list/tuple of segments
    :param proxy_weights: parameters of the instrumental distribution
    :param target_weights: parameters of the target distribution
    :param options: several options
    :return: a list of objects of the type Result
    """
    # creates a decoder
    decoder = cdeclib.create_decoder(cdec_config_str, proxy_weights)
    # prepares output
    return [sample(segment, n_samples, proxy_weights, target_weights, decoder) for segment in segments]


def write_to_file(result, odir, columns, sortby):
    """
    Write results to a file under `odir`
    :param result: importance samples
    :param columns: format to make string out of an importance sample
    :param odir: output directory
    """
    # log results
    try:
        header = '\t'.join('#{0}'.format(c) for c in columns)
        with open('{0}/{1}'.format(odir, result.segment.id), 'w') as out:
            print >> out, header
            for s in result.sorted(sortby):
                print >> out, s.format_str(columns)
            print >> out
    except:
        raise Exception('job={0} exception={1}'.format(result.segment.id,
                                                       ''.join(traceback.format_exception(*sys.exc_info()))))


def write_to_stdout(result, columns, sortby):
    """
    Write results to stdout
    :param results: importance samples
    :param columns: format to make string out of an importance sample
    :param odir: ignored
    """
    header = '#sid\t{0}'.format('\t'.join('#{0}'.format(c) for c in columns))
    print header
    for s in result.sorted(sortby):
        print '{0}\t{1}'.format(result.segment.id, s.format_str(columns))
    print


def sample_and_save(odir, columns, sortby, *args, **kwargs):
    write_to_file(sample(*args, **kwargs), odir, columns, sortby)


def argparse_and_config():
    parser = argparse.ArgumentParser(description='MC sampler for hiero models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config',
                        type=str, help='configuration file')
    parser.add_argument('workspace',
                        type=str, default=None,
                        help='samples will be written to $workspace/samples/$i')
    parser.add_argument("--target-scaling",
                        type=float, default=1.0,
                        help="scaling parameter for the target model (default: 1.0)")
    parser.add_argument("--proxy-scaling",
                        type=float, default=1.0,
                        help="scaling parameter for the proxy model (default: 1.0)")
    parser.add_argument("--samples",
                        type=int, default=100,
                        help="number of samples (default: 100)")
    parser.add_argument("--input-format",
                        type=str, default='cdec',
                        choices=['plain', 'cdec'],
                        help="'plain': one input sentence per line and requires --grammars; "
                             "'cdec': one input sentence per line plus a grammar path (sgml-formatted)")
    parser.add_argument("--grammars",
                        type=str,
                        help="where to find grammars (grammar files are expected to be named grammar.$i.sgm, "
                             "with $i 0-based)")
    parser.add_argument('--jobs', '-j',
                        type=int, default=2,
                        help='number of processes')
    parser.add_argument('--sortby',
                        type=str, default='none',
                        choices=['n', 'p', 'q', 'r', 'nr'],
                        help='sort results by a specific column')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='increases verbosity')

    args, config, failed = configure(parser,
                                     set_defaults=['chisel:model', 'chisel:sampler'],
                                     required_sections=['proxy', 'target', 'cdec'],
                                     configure_logging=True)
    logging.debug('arguments: %s', vars(args))

    # additional sanity checks: input format
    if args.input_format == 'plain' and args.grammars is None:
        logging.error("'--input-format plain' requires '--grammars <path>'")
        failed = True

    if failed:
        sys.exit(1)

    return args, config


def main():
    options, config = argparse_and_config()

    # make output dir
    output_dir = '{0}/samples'.format(options.workspace)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    logging.info('Writing samples to: %s', output_dir)

    # cdec configuration string
    cdec_cfg_string = cdeclib.make_cdec_config_string(config.items('cdec'), config.items('cdec:features'))
    logging.debug('cdec.ini: %s', repr(cdec_cfg_string))

    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')), options.proxy_scaling)
    logging.debug('proxy (scaling=%f): %s', options.proxy_scaling, dict2str(proxy_weights, sort=True))

    # parameters of the target distribution
    target_weights = scaled_fmap(section_literal_eval(config.items('target')), options.target_scaling)
    logging.debug('target (scaling=%f): %s', options.target_scaling, dict2str(target_weights, sort=True))

    # loads scorer modules
    if config.has_section('chisel:scorers'):
        scorers_map = section_literal_eval(config.items('chisel:scorers'))
        ff.load_scorers(scorers_map.itervalues())

    # scorers' configuration
    if config.has_section('chisel:scorers:config'):
        scorers_config = section_literal_eval(config.items('chisel:scorers:config'))
    else:
        scorers_config = {}
    logging.info('chisel:scorers:config: %s', scorers_config)

    # logs which features were added to the proxy
    extra_features = {k: v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.debug('Extra features: %s', extra_features)

    # configure scorers
    ff.configure_scorers(scorers_config)

    # reads segments from input
    segments = [SegmentMetaData.parse(line.strip(),
                                      options.input_format,
                                      sid=sid,
                                      grammar_dir=options.grammars)
                for sid, line in enumerate(sys.stdin)]

    logging.info('Distributing %d segments to %d jobs', len(segments), options.jobs)

    # log results
    columns = ('n', 'r', 'p', 'q', 'd', 'v')

    pool = Pool(options.jobs)
    # distribute jobs
    pool.map(partial(sample_and_save,
                     output_dir,
                     columns,
                     options.sortby,
                     n_samples=options.samples,
                     proxy_weights=proxy_weights,
                     target_weights=target_weights,
                     cdec_config_str=cdec_cfg_string),
             segments)


if __name__ == '__main__':
    main()