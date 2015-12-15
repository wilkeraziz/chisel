__author__ = 'waziz'

import re
from os.path import isfile, basename, splitext
import logging
from chisel.smt import Yield, Derivation, Tree, SVector
import math
from glob import glob
from collections import defaultdict, deque
from wmap import WMap
import gzip
from io import TextIOWrapper


def smart_ropen(path):
    """Open file in reading mode directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        #return TextIOWrapper(gzip.open(path, 'rb'))  # python3
        return gzip.open(path, 'rb')
    else:
        return open(path, 'r')


def smart_wopen(path):
    """Open file in writing mode directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        #return TextIOWrapper(gzip.open(path, 'wb'))  # python3
        return gzip.open(path, 'wb')
    else:
        return open(path, 'w')


# TODO: generalise format to be able to read lines with repeated keys
def read_config(path):
    """
    Read a config file made of key=value entries
    :param path: path to file
    :return: dictionary containing the key-value pairs in the config file
    """
    config = {}
    with smart_ropen(path) as f:
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
    with smart_ropen(path) as f:
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

    def to_sgm(self, dump_refs=True):
        if dump_refs and self.refs_:
            return '<seg grammar="{1}" id="{0}">{2}</seg> ||| {3}'.format(self.sid_,
                                                                          self.grammar_,
                                                                          self.src_,
                                                                          ' ||| '.join(str(ref) for ref in self.refs_))
        else:
            return '<seg grammar="{1}" id="{0}">{2}</seg>'.format(self.sid_,
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
            args['grammar'] = '{0}/grammar.{1}.gz'.format(grammar_dir, args['sid'])
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
            'sid': int(groups[1]),
            'src': Yield(groups[2]),
            'refs': [Yield(ref.strip()) for ref in parts[1:]]}


def parse_plain(plain_str):
    fields = plain_str.split(' ||| ')
    if len(fields) == 0:
        raise Exception('Missing fields: %s' % plain_str)
    args = {'src': Yield(fields[0])}
    if len(fields) > 1:
        args = {'refs': Yield(fields[1:])}
    return args


def list_numbered_files(basedir, sort=True, reverse=False):
    paths = glob('{0}/[0-9]*'.format(basedir))
    ids = [int(splitext(basename(path))[0]) for path in paths]
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


def read_sampled_derivations(iterable):
    """

    Read a file structed as follows

        [proxy]
        feature=value
        ...

        [target]
        feature=value
        ...

        [samples]
        # count projection vector
        ...
    """
    
    reserved = set(['[proxy]', '[target]', '[samples]'])

    def read_weights(Q):
        wmap = {}
        while Q:
            line = Q.popleft()
            if not line or line.startswith('#'):  # discarded lines
                continue
            if line in reserved:  # stop criterion
                Q.appendleft(line)
                break
            try:  # parse a pair
                k, v = line.split('=')
                wmap[k] = float(v)
            except:
                raise ValueError('Incorrectly formatted key-value pair: %s', line)
        return wmap
    
    def read_samples(Q):
        samples = []
        while Q:
            line = Q.popleft()
            if not line or line.startswith('#'):  # discarded lines
                continue
            if line in reserved:  # stop criterion
                Q.appendleft(line)
                break
            try:  # parse a row
                count, derivation, fmap = line.split('\t')
                d = Derivation(tree=Tree(derivationstr=derivation),
                               vector=SVector(fmap),
                               count=long(count))
                samples.append(d)
            except:
                raise ValueError('Incorrectly formatted sample: %s' % line)
        return samples

    Q = deque(line.strip() for line in iterable)
    qmap = {}
    pmap = {}
    samples = []
    while Q:
        line = Q.popleft()
        if not line or line.startswith('#'):
            continue
        if line == '[proxy]':
            qmap = read_weights(Q)
        elif line == '[target]':
            pmap = read_weights(Q)
        elif line == '[samples]':
            samples = read_samples(Q)
    return samples, WMap(qmap.iteritems()), WMap(pmap.iteritems())


def sampled_derivations_from_file(input_file):
    return read_sampled_derivations(smart_ropen(input_file))
