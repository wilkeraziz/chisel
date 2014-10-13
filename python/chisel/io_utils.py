"""
@author waziz
"""
import gzip
import re
from os.path import isfile

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

    def __init__(self, sid, src, grammar, tgt = '', alignment = []):
        self.sid_ = sid
        self.grammar_ = grammar
        self.src_ = src
        self.tgt_ = tgt
        self.alignment_ = tuple(alignment)

    @property
    def id(self):
        return self.sid_

    @property
    def src(self):
        return self.src_

    @property
    def grammar(self):
        return self.grammar_

    def __str__(self):
        return 'grammar=%s\tsrc=%s' % (self.grammar_, self.src_)

    @staticmethod
    def parse(sid, line, mode, grammar = None):
        # parse line
        if mode == 'plain':
            data = parse_plain(line)
        elif mode == 'cdec':
            data = parse_cdec_sgml(line)
        elif mode == 'moses':
            data = parse_moses_columns(line)
        elif mode == 'chisel':
            data = parse_chisel_columns(line)
        else:
            raise Exception('unknown input format: %s' % mode)
        # overrides
        if grammar is not None:
            data['grammar'] = '{0}/grammar.{1}.gz'.format(grammar, sid)
        # sanity checks
        if not isfile(data['grammar']):
            raise Exception('Grammar file not found: %s' % data['grammar'])
        # construct segment
        return SegmentMetaData(sid, data['src'], data['grammar'], data.get('tgt', ''), data.get('alignment', []))

def parse_plain(plain_str):
    fields = plain_str.split(' ||| ')
    return {'src':fields[0].strip()}

def parse_chisel_columns(chisel_str):
    columns = chisel_str.split('\t')
    if len(columns) < 2:
        raise Exception('missing fields: %s' % columns)
    return {'src':columns[1], 'grammar':columns[0]}

def parse_moses_columns(moses_str):
    columns = [token.strip() for token in moses_str.split('|||')]
    if len(columns) < 2:
        raise Exception('missing fields: %s' % columns)
    return {'src':columns[1], 'grammar':columns[0]}

def parse_cdec_sgml(sgml_str):
    """parses an sgml-formatted line as cdec would
    returns a dicts with grammar, id, src and tgt"""
    #pattern = re.compile('<seg grammar="([^"]+)" id="([0-9]+)">(.+)<\/seg> [|]{3} (.+)')
    pattern = re.compile('<seg grammar="([^"]+)" id="([0-9]+)">(.+)<\/seg>')
    match = pattern.match(sgml_str)
    groups = match.groups()
    if len(groups) < 3:
        raise Exception('missing fields: %s' % groups)
    return {'grammar':groups[0], 'sid':groups[1], 'src':groups[2]}

