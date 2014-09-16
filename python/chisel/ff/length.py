"""
Feature functions that capture information about the length of a complete hypothesis.

@author waziz
"""
import chisel

@chisel.ff.features('FLength', 'ELength', 'LengthRatio', 'LengthDiff')
def ELength(hypothesis): 
    f, e = hypothesis.source_.split(), hypothesis.translation_.split()
    return (len(f), len(e), float(len(e))/len(f), len(e) - len(f))

@chisel.ff.dense
def AbsLenDiff(hypothesis): 
    # we could use suffstats to avoid splitting text twice (in ELength and here)
    # however, split is pretty fast and I wanted to have an example using @ff.dense ;)
    f, e = hypothesis.source_.split(), hypothesis.translation_.split()
    return abs(len(e) - len(f))

