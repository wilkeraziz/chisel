"""
@author waziz
"""
import gzip
import re

def read_config(path):
    config = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            k, v = line.strip().split('=')
            config[k] = v
    return config

def read_weights(path, scaling = 1.0):
    weights = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            k, v = line.strip().split()
            weights[k] = float(v) * scaling
    return weights

def str2fmap(line):
    return {k:float(v) for k, v in (pair.split('=') for pair in line.split())}

def fmap2str(iterable):
    return ' '.join('%s=%s' % (k, str(v)) for k, v in iterable)

class SegmentMetaData(object):
    """
    A simple container for input segments
    """

    def __init__(self, src, grammar, tgt = '', alignment = []):
        self.grammar_ = grammar
        self.src_ = src
        self.tgt_ = tgt
        self.alignment_ = tuple(alignment)

    def __str__(self):
        return 'grammar=%s\tsrc=%s' % (self.grammar_, self.src_)

    @staticmethod
    def parse(line, mode):
        if mode == 'cdec':
            return parse_cdec_sgml(line)
        if mode == 'moses':
            return parse_moses_columns(line)
        if mode == 'chisel':
            return parse_chisel_columns(line)
        raise Exception('unknown input format: %s' % mode)

def parse_chisel_columns(chisel_str):
    columns = chisel_str.split('\t')
    if len(columns) < 2:
        raise Exception('missing fields: %s' % columns)
    return SegmentMetaData(src=columns[1], grammar=columns[0])

def parse_moses_columns(moses_str):
    columns = [token.strip() for token in moses_str.split('|||')]
    if len(columns) < 2:
        raise Exception('missing fields: %s' % columns)
    return SegmentMetaData(src=columns[1], grammar=columns[0])

def parse_cdec_sgml(sgml_str):
    """parses an sgml-formatted line as cdec would
    returns a dicts with grammar, id, src and tgt"""
    pattern = re.compile('<seg grammar="([^"]+)" id="([0-9]+)">(.+)<\/seg> [|]{3} (.+)')
    match = pattern.match(sgml_str)
    groups = match.groups()
    if len(groups) != 4:
        raise Exception('missing fields: %s' % groups)
    return SegmentMetaData(grammar = groups[0], src = groups[2], tgt = groups[3])

