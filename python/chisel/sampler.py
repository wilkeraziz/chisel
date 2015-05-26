"""
@author waziz
"""
import collections
import logging
import sys
import argparse
import math
import os
from time import time
from multiprocessing import Pool
from functools import partial
from ConfigParser import RawConfigParser
import ff
import cdeclib
import numpy as np
from util import fpairs2str, dict2str, fmap_dot, scaled_fmap
from util import resample as do_resample
from util.config import configure, section_literal_eval
from util.io import SegmentMetaData
import traceback
import itertools
from smt import SVector, Tree, Derivation
from instrumental import Sampler, KLOptimiser


class RawImportanceSample(object):
    """
    This is a container for the minimum necessary information about a sample
    """

    def __init__(self, derivation_str, count, fpairs, log_up, log_uq):
        """
        :param derivation_str: actual sample
        :param count: number of times it was sampled
        :param fpairs: pairs (fname, fvalue)
        :param log_up: target score (log of unnormalised target distribution)
        :param log_uq: proxy score (log of unnormalised instrumental distribution)
        """
        self.derivation_str_ = derivation_str
        self.count_ = count
        self.fpairs_ = fpairs
        self.log_up_ = log_up
        self.log_uq_ = log_uq

    @property
    def derivation_str(self):
        return self.derivation_str_

    @property
    def count(self):
        return self.count_

    @property
    def fpairs(self):
        return self.fpairs_

    @property
    def log_up(self):
        """returns ln(up(d))"""
        return self.log_up_

    @property
    def log_uq(self):
        """returns ln(uq(d))"""
        return self.log_uq_

    @property
    def log_ur(self):
        """returns ln(up(d)) - ln(uq(d))"""
        return self.log_up_ - self.log_uq_


class ImportanceSample(object):
    """
    This is a container for a sampled derivation
    along with basic estimates.
    """

    def __init__(self, derivation_str, count, log_up, log_uq, log_ur, importance, fpairs):
        """
        :param derivation_str: actual sample
        :param count: number of times it was sampled
        :param log_up: log of unnormalised p 
        :param log_uq: log of unnormalised q
        :param log_ur: log of unnormalised r
        :param importance: normalised importance (this is an estimate)
        :param fpairs: pairs (fname, fvalue)
        """
        self.derivation_str_ = derivation_str
        self.count_ = count
        self.log_up_ = log_up
        self.log_uq_ = log_uq
        self.log_ur_ = log_ur
        self.importance_ = importance
        self.fpairs_ = fpairs

    @property
    def derivation_str(self):
        return self.derivation_str_

    @property
    def projection(self):
        return self.derivation_str_

    @property
    def fpairs(self):
        return self.fpairs_

    @property
    def count(self):
        return self.count_
    
    @property
    def log_up(self):
        """(exact) log up(d)"""
        return self.log_up_

    @property
    def log_uq(self):
        """(exact) ln uq(d)"""
        return self.log_uq_

    @property
    def log_ur(self):
        """(exact) log ur(d)"""
        return self.log_ur_

    @property
    def importance(self):
        return self.importance_

    def __str__(self):
        return self.format_str()

    def format_str(self, keys='n importance log_up log_uq log_ur s v'.split(), separator='\t'):
        """
        Format as string
        :param keys: 
        :param separator:
        :return:
        """
        fields = [None] * len(keys)
        for i, k in enumerate(keys):
            if k == 'n' or k == 'count':
                x = self.count
            elif k == 'log_up':
                x = self.log_up
            elif k == 'log_uq':
                x = self.log_uq
            elif k == 'log_ur':
                x = self.log_ur
            elif k == 'importance':
                x = self.importance
            elif k == 's' or k == 'd':  # TODO: return derivation
                x = self.derivation_str
            elif k == 'v':
                x = fpairs2str(self.fpairs)
            else:
                raise Exception('Unkonwn field: %s' % k)
            fields[i] = str(x)
        return separator.join(fields)


class Result(object):

    def __init__(self, segment, raw_samples, resample=True):
        """
        A result associated a segment with its importance samples
        :param segment: the input segment
        :param samples: a list of ImportanceSample objects
        :param resample: if True resamples the importance (helps lower variance)
        """
        self.segment_ = segment
        
        # 1) dot products
        #q_dot = np.array([fmap_dot(d.vector, q_wmap) for d in raw_samples])
        #p_dot = np.array([fmap_dot(d.vector, p_wmap) for d in raw_samples])
        #r_dot = p_dot - q_dot

        # get total samples
        n_samples = sum(raw.count for raw in raw_samples)

        # estimate (normalised) q(d)
        log_uq = np.array([raw.log_uq + np.log(raw.count) for raw in raw_samples], float)
        log_Zq = np.logaddexp.reduce(log_uq)
        #log_q = log_uq - log_Zq
        #log_q2 = np.log(np.array([raw.count for raw in raw_samples], float) / n_samples)

        # estimate (normalised) importance weights
        log_ur = np.array([raw.log_ur + np.log(raw.count) for raw in raw_samples], float)
        log_Zr = np.logaddexp.reduce(log_ur)
        log_r = log_ur - log_Zr
        importance = np.exp(log_r)
        
        mean_r = importance.mean()
        mean_sr = np.exp(log_r * 2).mean()
        ne = importance.size * mean_r * mean_r / mean_sr

        if resample:  # importance resampling
            importance = do_resample(np.exp(log_r), n_samples)

        self.n_samples_ = n_samples
        #self.nonzero_ = np.array([raw_samples[i].count for i in np.nonzero(importance)[0]], float).sum() / n_samples
        self.Zratio_ = log_Zr - np.log(n_samples)
        self.ne_ = ne
        logging.debug('%d samples (ne=%f) Zp/Zq=%f', self.n_samples_, self.ne_, self.Zratio_)

        self.samples_ = [ImportanceSample(derivation_str=raw.derivation_str,
            count=raw.count,
            log_up=raw.log_up,
            log_uq=raw.log_uq,
            log_ur=raw.log_ur,
            importance=importance[i],
            fpairs=raw.fpairs) for i, raw in enumerate(raw_samples)]
       
        # TODO: delete list of ImportanceSamples and use instead Derivations
        self.derivations_ = [Derivation(tree=Tree(raw.derivation_str),
            vector=SVector(fpairs=raw.fpairs),
            count=raw.count,
            log_ur=raw.log_ur,
            importance=importance[i]) for i, raw in enumerate(raw_samples)]

    @property
    def segment(self):
        return self.segment_

    @property
    def samples(self):
        return self.samples_

    @property
    def n_samples(self):
        return self.n_samples_

    @property
    def Zratio(self):
        return self.Zratio_

    @property
    def ne(self):
        return self.ne_

    def n_derivations(self):
        return len(self.samples_)

    def n_strings(self):
        return len(frozenset(sample.derivation_str for sample in self.samples_))

    def sorted(self, opt):
        if opt == 'n':
            return sorted(self.samples_, key=lambda s: s.count, reverse=True)
        if opt == 'p':
            return sorted(self.samples_, key=lambda s: s.log_up, reverse=True)
        if opt == 'q':
            return sorted(self.samples_, key=lambda s: s.log_uq, reverse=True)
        if opt == 'r':
            return sorted(self.samples_, key=lambda s: s.importance, reverse=True)
        return iter(self.samples_)


def sample(segment, n_samples, resample, proxy_weights, target_weights, cdec_config_str='', decoder=None):
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
    forest = cdeclib.build_proxy(str(segment.src), segment.grammar, decoder)
    # samples from the proxy distribution
    q_samples = cdeclib.sample(forest, n_samples)
    # header = '\t'.join(['#count', '#translation', '#r', '#qmap', '#qdot', '#pmap', '#pdot'])
    # print header
    # for now we do not have access to alignment
    # ostream = [header]
    raw_samples = []
    for derivation_str, sample_info in sorted(q_samples.iteritems(), key=lambda pair: len(pair[1]), reverse=True):
        # computes additional features
        extraff = ff.compute_features(ff.Hypothesis(source=str(segment.src), translation=derivation_str))
        # groups vectors associated with equivalent derivations
        counter = collections.Counter(fpairs for fpairs, _ in sample_info)
        # compute target vectors
        # qdots, pdots = [], []
        for q_fpairs, count in counter.iteritems():
            # start with the features that are used in the proxy
            fmap = dict(q_fpairs)
            # include additional features (must not overwrite proxy features)
            for fname, fvalue in extraff:
                fmap[fname] = fvalue
            # target score (the dot might skip some features, it depends on target_weights)
            log_up = fmap_dot(fmap, target_weights)
            # instrumental score
            log_uq = fmap_dot(fmap, proxy_weights)
            # make output
            raw_samples.append(RawImportanceSample(derivation_str=derivation_str,
                                               count=count,
                                               fpairs=fmap.items(),
                                               log_up=log_up,
                                               log_uq=log_uq))

    """unnorm_r = np.array([x.log_r + np.log(x.count) for x in is_samples], float)
    total_r = np.logaddexp.reduce(unnorm_r)
    norm_r = unnorm_r - total_r
    importance = resample(np.exp(norm_r), n_samples)
    log_importance = np.log(importance)
    
    unnorm_p = np.array([x.log_p for x in is_samples], float)
    log_Zp = np.logaddexp.reduce(unnorm_p + log_importance)

    unnorm_q = np.array([x.log_q + np.log(x.count) for x in is_samples], float)
    total_q = np.logaddexp.reduce(unnorm_q)
    print '     Zq', total_q - np.log(n_samples)
    print '     Zp', log_Zp
    print 'nonzero', np.array([is_samples[i].count for i in np.nonzero(importance)[0]], float).sum()/n_samples"""

    # resets scorers to a null state
    ff.reset_scorers()
    return Result(segment, raw_samples, resample=resample)


def batch_sample(segments, n_samples, resample, cdec_config_str, proxy_weights, target_weights):
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
    return [sample(segment, n_samples, resample, proxy_weights, target_weights, decoder) for segment in segments]


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
        return result.segment.id, result.n_samples, result.n_derivations(), result.n_strings(), result.Zratio, result.ne
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
    try:
        return write_to_file(sample(*args, **kwargs), odir, columns, sortby)
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))





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
    parser.add_argument("--tuning-samples",
                        type=int, default=100,
                        help="number of samples when tuning the proxy (default: 100)")
    parser.add_argument("--resampling",
                        action='store_true',
                        help="resample the importance weights")
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
    parser.add_argument('--tune', '-T',
                        action='store_true',
                        help='Tune the proxy distribution')

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
                for sid, line in enumerate(sys.stdin)]  # easy to check variance (just need to multiply this by a number of trials) 

    logging.info('Distributing %d segments to %d jobs', len(segments), options.jobs)

    # log results
    columns ='n importance log_up log_uq log_ur d v'.split()
    #columns = ('n', 'r', 'p', 'q', 'd', 'v')


    # TODO run this in parallel
    for seg in segments:
        
        sampler = Sampler(seg, proxy_weights, target_weights, cdec_cfg_string)

        if options.tune:  # perhaps we tune Q by optimising KL(q||p)
            optimiser = KLOptimiser(seg, 
                    options.tuning_samples, 
                    proxy_weights, 
                    target_weights, 
                    cdec_cfg_string, 
                    avgcoeff=1.0)
            optq = optimiser.optimise()  # optimise the proxy
            sampler.reweight(optq)  # reweight the forest
        
        # samples
        samples = sampler.sample(options.samples)
        sampler.save(samples, output_dir)
        #TODO: update chisel.decision to deal with the clearner format (and with the tuned parameters)
        #TODO: update chisel.tuning if necessary


        #result = Result(seg, samples, options.resampling)
        #write_to_file(result, output_dir, columns, options.sortby)

    # done

    if False:  # TODO clean this up
        pool = Pool(options.jobs)
        # distribute jobs
        feedback = pool.map(partial(sample_and_save,
                                   output_dir,
                                   columns,
                                   options.sortby,
                                   n_samples=options.samples,
                                   resample=options.resampling, 
                                   proxy_weights=proxy_weights,
                                   target_weights=target_weights,
                                   cdec_config_str=cdec_cfg_string),
                           segments)

        # summary of samples
        try:
            from tabulate import tabulate
            print tabulate(feedback, headers=('job', 'samples', 'derivations', 'strings', 'Zp/Zq', 'Ne'), tablefmt='pipe')
        except ImportError:
            logging.info('Consider installing tabulate for some nice summaries')


if __name__ == '__main__':
    main()
