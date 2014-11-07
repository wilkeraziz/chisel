"""
FF example.

@author waziz
"""
# this is necessary!
import chisel

# here we import stuff we need for the computation of the feature
import random

# here we store some global stuff (e.g. models, parameters, etc)
interval = None
src_parse = None
tgt_parse = None

# here we configure the feature extractor, that is, load models, parse user-set parameters, etc
@chisel.ff.configure
def configure(config):
    global interval
    if 'dummy.a' not in config:
        raise Exception('Perhaps you forgot to configure `dummy.a=int` in your chisel.ini file?')
    if 'dummy.b' not in config:
        raise Exception('Perhaps you forgot to configure `dummy.b=int` in your chisel.ini file?')
    a, b = config['dummy.a'], config['dummy.b']
    interval = (a, b)

# preprocessing and cleaning (wrt source)

@chisel.ff.preprocess
def preprocess_src(segment):
    global src_parse
    src_parse = segment.src.split()

@chisel.ff.reset
def reset():
    global src_parse
    src_parse = None

# preprocessing and cleaning (wrt target)

@chisel.ff.suffstats
def preprocess_tgt(hypo):
    global tgt_parse
    tgt_parse = hypo.tgt.split()

@chisel.ff.cleanup
def cleanup():
    global tgt_parse
    tgt_parse = None

# feature definitions

# this feature will be named Dummy0
@chisel.ff.dense
def Dummy0(hypothesis): 
    return random.uniform(interval[0], interval[1]) * len(tgt_parse)

# these features will be named YetAnotherDummy1 and YetAnotherDummy2
@chisel.ff.features('YetAnotherDummy1', 'YetAnotherDummy2')
def Dummy2(hypothesis): 
    return (len(src_parse) * random.uniform(interval[0], interval[1]), len(tgt_parse) * random.uniform(interval[0], interval[1]))

# these feature will be named Dummy_<suffix> 
@chisel.ff.sparse
def Dummy(hypothesis): 
    return ('v1', 1), ('v2', 0.5), ('v5', -1)
    
if __name__ == '__main__':
    print __doc__
