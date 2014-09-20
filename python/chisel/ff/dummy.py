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

# here we configure the feature extractor, that is, load models, parse user-set parameters, etc
@chisel.ff.configure
def configure(config):
    global interval
    conf_str = config['Dummy'] 
    a, b = [int(x) for x in conf_str.split()]
    interval = (a, b)

# here we define feature functions

# this feature will be named Dummy0
@chisel.ff.dense
def Dummy0(hypothesis): 
    return random.uniform(interval[0], interval[1]) * len(hypothesis.translation_.split())

# these features will be named YetAnotherDummy1 and YetAnotherDummy2
@chisel.ff.features('YetAnotherDummy1', 'YetAnotherDummy2')
def Dummy2(hypothesis): 
    return (len(hypothesis.translation_.split()), random.uniform(interval[0], interval[1]))

# these feature will be named Dummy_<suffix> 
@chisel.ff.sparse
def Dummy(hypothesis): 
    return ('v1', 1), ('v2', 0.5), ('v5', -1)
    
if __name__ == '__main__':
    print __doc__
