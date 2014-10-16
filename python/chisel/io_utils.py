__author__ = 'waziz'

import re
from os.path import isfile, basename
import logging
from smt import Derivation, Tree, SVector
import math
from glob import glob


# TODO: generalise format to be able to read lines with repeated keys
def read_config(path):
    """
    Read a config file made of key=value entries
    :param path: path to file
    :return: dictionary containing the key-value pairs in the config file
    """
    config = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            k, v = line.strip().split('=')
            config[k] = v
    return config


# TODO: generalise format to be robust to little differences (e.g. FName=FWeight)
def read_weights(path, scaling=1.0):
    """
    Read a file made of `FName FWeight` entries
    :param path:
    :param scaling:
    :return:
    """
    weights = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            k, v = line.strip().split()
            weights[k] = float(v) * scaling
    return weights


class SegmentMetaData(object):
    """
    A simple container for input segments
    """

    def __init__(self, sid, src, grammar, tgt='', alignment=[]):
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
    def parse(sid, line, mode, grammar=None):
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
    return {'src': fields[0].strip()}


def parse_chisel_columns(chisel_str):
    columns = chisel_str.split('\t')
    if len(columns) < 2:
        raise Exception('missing fields: %s' % columns)
    return {'src': columns[1], 'grammar': columns[0]}


def parse_moses_columns(moses_str):
    columns = [token.strip() for token in moses_str.split('|||')]
    if len(columns) < 2:
        raise Exception('missing fields: %s' % columns)
    return {'src': columns[1], 'grammar': columns[0]}


def parse_cdec_sgml(sgml_str):
    """parses an sgml-formatted line as cdec would
    returns a dicts with grammar, id, src and tgt"""
    #pattern = re.compile('<seg grammar="([^"]+)" id="([0-9]+)">(.+)<\/seg> [|]{3} (.+)')
    pattern = re.compile('<seg grammar="([^"]+)" id="([0-9]+)">(.+)<\/seg>')
    match = pattern.match(sgml_str)
    groups = match.groups()
    if len(groups) < 3:
        raise Exception('missing fields: %s' % groups)
    return {'grammar': groups[0], 'sid': groups[1], 'src': groups[2]}


def list_numbered_files(basedir, sort=True, reverse=False):
    paths = glob('{0}/[0-9]*'.format(basedir))
    ids = [int(basename(path)) for path in paths]
    if not sort:
        return zip(ids, paths)
    else:
        return sorted(zip(ids, paths), key=lambda pair: pair[0], reverse=reverse)


def next_block(fi):
    """Yields the next block of non-empty lines from an input stream (stripped lines are returned in a list)"""
    block = []
    for line in fi:
        line = line.strip()
        if not line and len(block):
            yield block
            block = []
        else:
            block.append(line)
    if len(block):
        yield block


def read_block(fi):
    """Read and returns one block of non-empty lines from an input stream (stripped lines are returned in a list)"""
    block = []
    for line in fi:
        line = line.strip()
        if not line and len(block):
            break
        block.append(line)
    return block


def read_sampled_derivations(iterable, required=dict((k, k) for k in 'derivation:d vector:v score:p count:n importance:r'.split())):
    """
    Parse a file containing sampled derivations. The file is structured as a table (tab-separated columns).
    The first line contains the column names.
    We expect at least a fixed set of columns (e.g. derivation, vector, score, count, importance).
    The table must be grouped by derivation.
    @return list of solutions
    """
    logging.info('reading from %s', iterable)
    # get the column names
    raw = next(iterable)
    if not raw.startswith('#'):
        raise Exception('missing header')
    colnames = [colname.replace('#', '') for colname in raw.strip().split('\t')]
    needed = frozenset(required.itervalues())
    # sanity check
    if not (needed <= frozenset(colnames)):
        raise Exception('missing columns: %s' % ', '.join(needed - frozenset(colnames)))
    logging.info('%d columns: %s', len(colnames), colnames)
    # parse rows
    D = []
    for row in (raw.strip().split('\t') for raw in iterable):
        k2v = {key: value for key, value in zip(colnames, row)}
        d = Derivation(tree=Tree(k2v[required['derivation']]),
                       vector=SVector(k2v[required['vector']]),
                       score=float(k2v[required['score']]),
                       count=int(k2v[required['count']]),
                       importance=math.exp(float(k2v[required['importance']])))
        D.append(d)
    logging.info('%d rows', len(D))
    return D