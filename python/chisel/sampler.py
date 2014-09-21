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

    @property
    def src(self):
        return self.source_

    @property
    def tgt(self):
        return self.translation_

def map_dot(fmap, wmap):
    return sum(fmap.get(fname, 0) * fweight for fname, fweight in wmap.iteritems())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'MC sampler for hiero models')
    parser.add_argument("proxy", type=str, help="feature weights (proxy model)")
    parser.add_argument("target", type=str, help="feature weights (target model)")
    parser.add_argument("chisel", type=str, help="chisel's config file")
    parser.add_argument("cdec", type=str, help="cdec's config file")
    parser.add_argument("--scaling", type=float, default = 1.0, help = "scaling parameter for the model (default: 1.0)") 
    parser.add_argument("--samples", type=int, default = 100, help = "number of samples (default: 100)") 
    parser.add_argument('-f', "--features", action='append', default = [], help = "additional feature definitions") 
    parser.add_argument("--input-format", type=str, default='plain', help="'plain': one input sentence per line and requires --grammars (default option); 'chisel': tab-separated columns [grammar source]; 'cdec': sgml-formatted; 'moses': |||-separated columns [grammar source]")
    parser.add_argument("--grammars", type=str, help = "where to find grammars (grammar files are expected to be named grammar.$i.sgm, with $i 0-based)") 
    #parser.add_argument("--top", type=int, default = 10, help = "Top n MBR solutions") 
    options = parser.parse_args()

    # sanity checks
    if options.input_format == 'plain' and options.grammars is None:
        logging.error("'--input-format plain' requires '--grammars <path>'")
        sys.exit()

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s %(message)s') 
    
    proxy_weights = read_weights(options.proxy, options.scaling)
    target_weights = read_weights(options.target, options.scaling)
    extra_features = {k:v for k, v in target_weights.iteritems() if k not in proxy_weights}
    logging.info('Extra features: %s', extra_features)

    config = read_config(options.chisel)
    logging.info('chisel.ini: %s', config)
    # loads scorer modules 
    ff.load_scorers(options.features)
    # configures scorer modules
    ff.configure_scorers(config)
    # reads from input
    for sid, line in enumerate(sys.stdin):
        # parses input segment
        segment = SegmentMetaData.parse(sid, line.strip(), options.input_format, options.grammars)
        logging.info('Translating: %s', segment.src)
        # pre-process the input (some scorers might require analysis of the input segment)
        ff.preprocess_input(segment)
        # builds the proxy distribution
        forest = build_proxy(segment.src, segment.grammar, options.proxy, options.scaling, options.cdec)
        # samples from the proxy distribution
        samples = sample(forest, options.samples)
        header = '\t'.join(['#count', '#translation', '#r', '#qmap', '#qdot', '#pmap', '#pdot'])
        print header
        # for now we do not have access to alignment  
        ostream = [header]
        for sample_str, sample_info in sorted(samples.iteritems(), key = lambda pair : len(pair[1]), reverse = True):
            #print >> sys.stderr, len(sample_info), sample_str
            # computes additional features
            extraff = ff.compute_features(Hypothesis(source = segment.src, translation = sample_str))
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
        # resets scorers to a null state
        ff.reset_scorers()
    
    
    #alternative = {'derivation':'translation', 'vector':'pmap', 'score':'r', 'count':'count'}
    #solutions = decision.read_solutions(iter(ostream), alternative)
    #decision.importance_sampling(solutions, options.top, importance = lambda sample : sample.normscore)
