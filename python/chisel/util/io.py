__author__ = 'waziz'

import re
from os.path import isfile, basename
import logging
from chisel.smt import Derivation, Tree, SVector
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

    def __init__(self, sid, src, grammar, refs=[]):
        self.sid_ = sid
        self.grammar_ = grammar
        self.src_ = src
        self.refs_ = tuple(refs)

    @property
    def id(self):
        return self.sid_

    @property
    def src(self):
        return self.src_

    @property
    def refs(self):
        return self.refs_

    @property
    def grammar(self):
        return self.grammar_

    def __str__(self):
        return 'grammar=%s\tsrc=%s' % (self.grammar_, self.src_)

    def to_sgm(self):
        if self.refs_:
            return '<seg grammar="{0}" id="{1}">{2}</seg> ||| {3}'.format(self.sid_,
                                                                          self.grammar_,
                                                                          self.src_,
                                                                          ' ||| '.join(self.refs_))
        else:
            return '<seg grammar="{0}" id="{1}">{2}</seg>'.format(self.sid_,
                                                                  self.grammar_,
                                                                  self.src_)

    @staticmethod
    def parse(line, mode, sid=None, grammar_dir=None):
        # parse line
        if mode == 'plain':
            args = parse_plain(line)
        if mode == 'cdec':
            args = parse_cdec_sgml(line)
        else:
            raise Exception('unknown input format: %s' % mode)
        # overrides sentence id
        if sid is not None:
            args['sid'] = sid
        # overrides grammar
        if grammar_dir is not None:
            args['grammar'] = '{0}/grammar.{1}.gz'.format(grammar_dir, sid)
        # sanity checks
        if not isfile(args['grammar']):
            raise Exception('Grammar file not found: %s' % args['grammar'])
        # construct segment
        return SegmentMetaData(**args)


def parse_cdec_sgml(sgml_str):
    """parses an sgml-formatted line as cdec would
    returns a dicts with grammar, id, src and tgt"""
    parts = sgml_str.split(' ||| ')
    if not parts:
        raise Exception('Missing fields' % sgml_str)
    pattern = re.compile('<seg grammar="([^"]+)" id="([0-9]+)">(.+)<\/seg>')
    match = pattern.match(parts[0])
    if match is None:
        raise Exception('Bad sgml: %s' % parts[0])
    groups = match.groups()
    return {'grammar': groups[0],
            'sid': groups[1],
            'src': groups[2],
            'refs': [ref.strip() for ref in parts[1:]]}


def parse_plain(plain_str):
    fields = plain_str.split(' ||| ')
    if len(fields) == 0:
        raise Exception('Missing fields: %s' % plain_str)
    args = {'src': fields[0]}
    if len(fields) > 1:
        args = {'refs': fields[1:]}
    return args


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


def read_sampled_derivations(iterable, required=dict(
    (k, k) for k in 'derivation:d vector:v score:p count:n importance:r'.split())):
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