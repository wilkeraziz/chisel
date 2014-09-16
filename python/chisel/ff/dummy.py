"""
FF example.

@author waziz
"""
# this is necessary!
import ff

# here we import stuff we need for the computation of the feature
import random

# here we store some global stuff (e.g. models, parameters, etc)
interval = None

# here we configure the feature extractor, that is, load models, parse user-set parameters, etc
@ff.configure
def configure(config):
    global interval
    conf_str = config['Dummy'] 
    a, b = [int(x) for x in conf_str.split()]
    interval = (a, b)

# here we define feature functions

# this feature will be named Dummy0
@ff.feature
def Dummy0(hypothesis): 
    return random.uniform(interval[0], interval[1]) * len(hypothesis.translation_.split())

# this feature will be named AnotherDummy
@ff.features('AnotherDummy')
def Dummy1(hypothesis): 
    return - random.uniform(interval[0], interval[1]) * len(hypothesis.translation_.split())

# these features will be named YetAnotherDummy1 and YetAnotherDummy2
@ff.features('YetAnotherDummy1', 'YetAnotherDummy2')
def Dummy2(hypothesis): 
    return (len(hypothesis.translation_.split()), random.uniform(interval[0], interval[1]))


