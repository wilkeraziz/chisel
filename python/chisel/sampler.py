"""
@author waziz
"""
import collections
import logging
import sys
import argparse
import math
import os

import ff
import cdeclib
from io_utils import read_config, read_weights, SegmentMetaData, fmap2str
from multiprocessing import Pool


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
        return '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(self.count,
                                                     self.r,
                                                     self.p,
                                                     self.q,
                                                     self.sample_str,
                                                     fmap2str(self.fpairs))

    @staticmethod
    def header():
        return 'count', 'log_ur', 'log_up', 'log_uq', 'sample', 'fpairs'


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
            return sorted(self.samples_, key = lambda sample : sample.count, reverse=True)
        if opt == 'p':
            return sorted(self.samples_, key = lambda sample : sample.p, reverse=True)
        if opt == 'q':
            return sorted(self.samples_, key = lambda sample : sample.q, reverse=True)
        if opt == 'r':
            return sorted(self.samples_, key = lambda sample : sample.r, reverse=True)
        if opt == 'nr':
            return sorted(self.samples_, key = lambda sample : sample.count * math.exp(sample.r), reverse=True)

        return iter(self.samples_)

def map_dot(fmap, wmap):
    return sum(fmap.get(fname, 0) * fweight for fname, fweight in wmap.iteritems())


def sample(segment, proxy_weights, target_weights, options, decoder = None):
    """
    Sample translation derivations for a given segment.
    :param segment: segment to be translated
    :param proxy_weights: parameters of the instrumental distribution
    :param target_weights: parameters of the target distribution
    :param options: several options
    :param decoder: a possibly null cdec.Decoder object
    :return: an instance of Result
    """
    # creates a decoder if necessary
    if decoder is None:
        decoder = cdeclib.create_decoder(options.cdec, options.proxy, options.scaling)

    logging.info('Translating (%d): %s', segment.id, segment.src)
    # pre-process the input (some scorers might require analysis of the input segment)
    ff.preprocess_input(segment)
    # builds the proxy distribution
    forest = cdeclib.build_proxy(segment.src, segment.grammar, decoder)
    # samples from the proxy distribution
    q_samples = cdeclib.sample(forest, options.samples)
    #header = '\t'.join(['#count', '#translation', '#r', '#qmap', '#qdot', '#pmap', '#pdot'])
    #print header
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
            pdot = map_dot(fmap, target_weights)
            # proxy score
            qdot = map_dot(fmap, proxy_weights)
            # make output
            is_samples.append(ImportanceSample(sample_str,
                                      count,
                                      fmap.items(),
                                      pdot,
                                      pdot - qdot))
    # resets scorers to a null state
    ff.reset_scorers()
    return Result(segment, is_samples)


def batch_sample(segments, proxy_weights, target_weights, options):
    """
    As :func:`sample`, however processes a batch of segments
    :param segments: list/tuple of segments
    :param proxy_weights: parameters of the instrumental distribution
    :param target_weights: parameters of the target distribution
    :param options: several options
    :return: a list of objects of the type Result
    """
    # creates a decoder
    decoder = cdeclib.create_decoder(options.cdec, options.proxy, options.scaling)
    # prepares output
    return [sample(segment, proxy_weights, target_weights, options, decoder) for segment in segments]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MC sampler for hiero models')
    parser.add_argument("proxy", type=str, help="feature weights (proxy model)")
    parser.add_argument("target", type=str, help="feature weights (target model)")
    parser.add_argument("chisel", type=str, help="chisel's config file")
    parser.add_argument("cdec", type=str, help="cdec's config file")
    parser.add_argument("--scaling", type=float, default=1.0, help="scaling parameter for the model (default: 1.0)")
    parser.add_argument("--samples", type=int, default=100, help="number of samples (default: 100)")
    parser.add_argument('-f', "--features", action='append', default=[], help="additional feature definitions")
    parser.add_argument("--input-format", type=str, default='plain',
                        help="'plain': one input sentence per line and requires --grammars (default option); 'chisel': tab-separated columns [grammar source]; 'cdec': sgml-formatted; 'moses': |||-separated columns [grammar source]")
    parser.add_argument("--grammars", type=str,
                        help="where to find grammars (grammar files are expected to be named grammar.$i.sgm, with $i 0-based)")
    # parser.add_argument("--top", type=int, default = 10, help = "Top n MBR solutions")
    parser.add_argument('--jobs', type=int, default=2, help='number of processes')
    parser.add_argument('--sortby', type=str, default='none', help='sort results by one of {n, p, q, r, nr}')
    parser.add_argument('--odir', type=str, default='', help='output dir')
    options = parser.parse_args()

    # sanity checks
    if options.input_format == 'plain' and options.grammars is None:
        logging.error("'--input-format plain' requires '--grammars <path>'")
        sys.exit()

    if options.odir:
        if not os.path.isdir(options.odir):
            os.mkdir(options.odir)
        logging.info('Writing output to %s', options.odir)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # parameters of the instrumental distribution
    proxy_weights = read_weights(options.proxy, options.scaling)
    # parameters of the target distribution
    target_weights = read_weights(options.target, options.scaling)
    # find out additional features
    extra_features = {k: v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.info('Extra features: %s', extra_features)
    # config file
    config = read_config(options.chisel)
    logging.info('chisel.ini: %s', config)
    # loads scorer modules
    ff.load_scorers(options.features)
    # configures scorer modules
    ff.configure_scorers(config)

    # reads segments from input
    segments = [SegmentMetaData.parse(sid,
                                      line.strip(),
                                      options.input_format,
                                      options.grammars)
                for sid, line in enumerate(sys.stdin)]

    logging.info('Distributing %d segments to %d jobs', len(segments), options.jobs)

    def sampling_wrapper(seg):
        return sample(seg, proxy_weights, target_weights, options)

    # distribute jobs
    pool = Pool(options.jobs)
    results = pool.map(sampling_wrapper, segments)

    # log results
    header = '\t'.join('#{0}'.format(x) for x in ImportanceSample.header())
    for result in results:
        # get output stream
        if options.odir:
            fout = open('{0}/samples.{1}'.format(options.odir, result.segment.id), 'w')
        else:
            fout = sys.stdout
        print >> fout, '#sid={0}'.format(result.segment.id)
        print >> fout, header
        for sample in result.sorted(options.sortby):
            print >> fout, sample
        print >> fout


    # alternative = {'derivation':'translation', 'vector':'pmap', 'score':'r', 'count':'count'}
    #solutions = decision.read_solutions(iter(ostream), alternative)
    #decision.importance_sampling(solutions, options.top, importance = lambda sample : sample.normscore)
