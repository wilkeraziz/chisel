"""
@author waziz
"""
import collections
import logging
import os
import sys
import argparse
import cdec
import gzip
import numpy as np
import re
import ff 
import math
#import decision

from io_utils import read_config, read_weights, SegmentMetaData, fmap2str, str2fmap

def build_proxy(input_str, grammar_file, weights_file, scaling, cdec_ini):
   
    with open(cdec_ini) as f:
        config_str = f.read()
        logging.info('cdec.ini:\n\t%s', '\n\t'.join(config_str.strip().split('\n')))
        # perhaps make sure formalism=scfg and intersection_strategy=full?
        #decoder = cdec.Decoder(config_str=config_str, formalism='scfg', intersection_strategy='Full')
        decoder = cdec.Decoder(config_str=config_str)

    logging.info('Loading weights: %s', weights_file)
    decoder.read_weights(weights_file, scaling)
    #logging.info('Weights: %s', dict(decoder.weights))
    
    logging.info('Loading grammar: %s', grammar_file)
    with gzip.open(grammar_file) as f:
        grammar = f.read()

    logging.info('Composing the forest')
    forest = decoder.translate(input_str, grammar = grammar)
    return forest

def sample(forest, n):
    sampledict = collections.defaultdict(list)
    for sample_str, sample_dot, sample_fmap in forest.sample_hypotheses(n):
        sampledict[sample_str.encode('utf8')].append((dict(sample_fmap), sample_dot))
    return sampledict

class Hypothesis(object):

    def __init__(self, source, translation):
        self.source_ = source
        self.translation_ = translation

def map_dot(fmap, wmap):
    return sum(fmap.get(fname, 0) * fweight for fname, fweight in wmap.iteritems())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'MC sampler for hiero models')
    parser.add_argument("proxy", type=str, help="feature weights (proxy model)")
    parser.add_argument("target", type=str, help="feature weights (target model)")
    parser.add_argument("chisel", type=str, help="chisel's config file")
    parser.add_argument("cdec", type=str, help="cdec's config file")
    parser.add_argument("--scaling", type=float, default = 1.0, help = "scaling parameter for the model") 
    parser.add_argument("--samples", type=int, default = 100, help = "number of samples") 
    parser.add_argument("--top", type=int, default = 10, help = "Top n MBR solutions") 
    parser.add_argument('-f', "--features", action='append', default = [], help = "additional feature definitions") 
    parser.add_argument("--input-format", type=str, default='chisel', help="chisel (tab-separated columns: grammar source), cdec (sgml), moses (|||-separated columns: grammar source)")
    options = parser.parse_args()

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s %(message)s') 
    
    proxy_weights = read_weights(options.proxy, options.scaling)
    target_weights = read_weights(options.target, options.scaling)
    extra_features = {k:v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.info('Extra features: %s', extra_features)

    config = read_config(options.chisel)
    logging.info('chisel.ini: %s', config)
    ff.load_features(options.features)
    resources = ff.configure_features(config)

    for line in sys.stdin:
        # parses input format
        segment = SegmentMetaData.parse(line.strip(), options.input_format)
        # builds the proxy distribution
        forest = build_proxy(segment.src_, segment.grammar_, options.proxy, options.scaling, options.cdec)
        # samples from the proxy distribution
        samples = sample(forest, options.samples)
        header = '\t'.join(['#count', '#translation', '#r', '#qmap', '#qdot', '#pmap', '#pdot'])
        print header
        # for now we do not have access to alignment  
        ostream = [header]
        for sample_str, sample_info in sorted(samples.iteritems(), key = lambda pair : len(pair[1]), reverse = True):
            #print >> sys.stderr, len(sample_info), sample_str
            # computes additional features
            extraff = ff.compute_features(Hypothesis(source = segment.src_, translation = sample_str))
            # groups vectors associated with equivalent derivations
            counter = collections.Counter(frozenset(fmap.iteritems()) for fmap, _ in sample_info)
            # compute target vectors
            qdots, pdots = [], []
            for fpairs, count in counter.iteritems():
                # features that are reused from the proxy
                qmap = dict(fpairs)
                pmap = dict(fpairs)
                # additional features (can overwrite proxy features)
                for fname, fvalue in extraff:
                    pmap[fname] = fvalue
                # target score (the dot might skip some features, it depends on target_weights)
                pdot = map_dot(pmap, target_weights)
                # proxy score
                qdot = map_dot(qmap, proxy_weights)
                # output info
                output = [str(count), 
                        sample_str,
                        str(math.exp(pdot - qdot)),
                        fmap2str(fpairs),
                        str(qdot),
                        fmap2str(pmap.iteritems()),
                        str(pdot)]
                ostream.append('\t'.join(output))
                print ostream[-1]
                qdots.append(qdot)
                pdots.append(pdot)
    
    
    #alternative = {'derivation':'translation', 'vector':'pmap', 'score':'r', 'count':'count'}
    #solutions = decision.read_solutions(iter(ostream), alternative)
    #decision.importance_sampling(solutions, options.top, importance = lambda sample : sample.normscore)
