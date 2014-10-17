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
from util import fpairs2str, dict2str, fmap_dot, scaled_fmap, section_literal_eval
from util.io import SegmentMetaData


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
    #ostream = [header]
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
    header = '\t'.join('#{0}'.format(c) for c in columns)
    with open('{0}/{1}'.format(odir, result.segment.id), 'w') as out:
        print >> out, header
        for s in result.sorted(sortby):
            print >> out, s.format_str(columns)
        print >> out


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
    conf_parser = argparse.ArgumentParser(description='MC sampler for hiero models',
                                          add_help=False)
    # first we deal with general configuration
    conf_parser.add_argument("config", type=str, help="config file")
    args, remaining_argv = conf_parser.parse_known_args()
    # parse the config file
    #config = SimpleConfigParser()
    config = RawConfigParser()
    # this is necessary in order not to lowercase the keys
    config.optionxform = str
    config.read(args.config)
    # some command line options may be overwritten by the section 'chisel:sampler' in the config file
    sampler_options = section_literal_eval(config.items('chisel:sampler'))

    # now we add specific options
    parser = argparse.ArgumentParser(description='MC sampler for hiero models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     parents=[conf_parser])

    parser.add_argument('--workspace', type=str, default=None,
                        help='samples will be written to $workspace/samples/$i')
    parser.add_argument("--scaling", type=float, default=1.0, help="scaling parameter for the model (default: 1.0)")
    parser.add_argument("--samples", type=int, default=100, help="number of samples (default: 100)")
    parser.add_argument("--input-format", type=str, default='plain',
                        choices=['plain', 'chisel', 'moses', 'cdec'],
                        help="'plain': one input sentence per line and requires --grammars (default option); "
                             "'chisel': tab-separated columns [grammar source]; 'cdec': sgml-formatted; "
                             "'moses': |||-separated columns [grammar source]")
    parser.add_argument("--grammars", type=str,
                        help="where to find grammars (grammar files are expected to be named grammar.$i.sgm, "
                             "with $i 0-based)")
    parser.add_argument('--jobs', type=int, default=2, help='number of processes')
    parser.add_argument('--sortby', type=str, default='none',
                        choices=['n', 'p', 'q', 'r', 'nr'],
                        help='sort results by a specific column')

    #parser.add_argument("--proxy", type=str, help="feature weights (proxy model)")
    #parser.add_argument("--target", type=str, help="feature weights (target model)")
    #parser.add_argument("--cdec", type=str, help="cdec's config file")
    #parser.add_argument("--resources", type=str, help="external resources config file")

    parser.set_defaults(**sampler_options)
    args = parser.parse_args()

    # overwrite global config file using command line specific config files
    #if args.proxy:
    #    config.replace('proxy', args.proxy)
    #if args.target:
    #    config.replace('target', args.target)
    #if args.cdec:
    #    config.replace('cdec', args.cdec)
    #if args.resources:
    #    config.replace('chisel:sampler:resources', args.resources)

    # start logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # sanity checks
    failed = False

    # individual configurations
    if not config.has_section('proxy'):
        logging.error("add a [proxy] section to the config file or provide --proxy")
        failed = True
    if not config.has_section('target'):
        logging.error("add a [target] section to the config file or provide --target")
        failed = True
    if not config.has_section('cdec'):
        logging.error("add a [cdec] section to the config file or provide --cdec")
        failed = True
    # input format
    if args.input_format == 'plain' and args.grammars is None:
        logging.error("'--input-format plain' requires '--grammars <path>'")
        failed = True

    if failed:
        sys.exit(1)

    return args, config


def main():
    options, config = argparse_and_config()

    output_dir = None
    if options.workspace:
        output_dir = '{0}/samples'.format(options.workspace)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        logging.info('Writing output to: %s', output_dir)

    # cdec configuration string
    cdec_cfg_string = cdeclib.make_cdec_config_string(config.items('cdec'), config.items('cdec:features'))
    logging.info('cdec.ini:\n%s', cdec_cfg_string)

    # parameters of the instrumental distribution
    proxy_weights = scaled_fmap(section_literal_eval(config.items('proxy')), options.scaling)
    logging.info('proxy: %s', dict2str(proxy_weights, sort=True))

    # parameters of the target distribution
    target_weights = scaled_fmap(section_literal_eval(config.items('target')), options.scaling)
    logging.info('target: %s', dict2str(target_weights, sort=True))

    # loads scorer modules
    if config.has_section('chisel:scorers'):
        scorers_map = section_literal_eval(config.items('chisel:scorers'))
        ff.load_scorers(scorers_map.itervalues())

    # load resources
    if config.has_section('chisel:resources'):
        resources = section_literal_eval(config.items('chisel:resources'))
    else:
        resources = {}
    logging.info('chisel:resources: %s', resources)

    # logs which features were added to the proxy
    extra_features = {k: v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.info('Extra features: %s', extra_features)

    # configure scorers
    ff.configure_scorers(resources)

    # reads segments from input
    segments = [SegmentMetaData.parse(sid,
                                      line.strip(),
                                      options.input_format,
                                      options.grammars)
                for sid, line in enumerate(sys.stdin)]

    logging.info('Distributing %d segments to %d jobs', len(segments), options.jobs)

    # log results
    columns = ('n', 'r', 'p', 'q', 'd', 'v')

    # for seg in segments:
    #    write_to_stdout(sampling_wrapper_with_return(seg),
    #                    columns,
    #                    options.sortby)
    #sys.exit(0)

    if output_dir is None:
        pool = Pool(options.jobs)
        # distribute jobs
        results = pool.map(partial(sample,
                                   n_samples=options.samples,
                                   proxy_weights=proxy_weights,
                                   target_weights=target_weights,
                                   cdec_config_str=cdec_cfg_string),
                           segments)
        # write to stdout after completing jobs
        [write_to_stdout(result, columns, options.sortby) for result in results]
    else:
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

        # alternative = {'derivation':'translation', 'vector':'pmap', 'score':'r', 'count':'count'}
        #solutions = decision.read_solutions(iter(ostream), alternative)
        #decision.importance_sampling(solutions, options.top, importance = lambda sample : sample.normscore)


if __name__ == '__main__':
    main()